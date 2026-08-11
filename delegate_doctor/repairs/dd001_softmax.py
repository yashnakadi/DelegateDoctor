"""DD-001 - softmax on a dimension other than the last.

The problem
-----------
ExecuTorch's XNNPACK partitioner will only delegate a softmax whose target
dimension is the last dimension of its input. The rule lives in
`backends/xnnpack/partition/config/generic_node_configs.py`, in
`SoftmaxConfig.check_constraints`, and reads (paraphrased):

    if not (dim == -1 or dim == tensor_dims - 1):
        reject: "dim must be the last dim"

A softmax on any other axis is refused and runs on ExecuTorch's portable
reference kernel, which is single-threaded. It is also usually a terrible
memory access pattern: for a (1, 21, 256, 256) tensor with dim=1, each of the
65 536 softmax vectors has 21 members that are 65 536 elements (256 KB) apart,
so effectively every element access is a cache miss.

This shape is extremely common - it is how a segmentation model turns
(N, classes, H, W) logits into probabilities.

The repair
----------
Move the softmax axis to the end, softmax there, and move it back:

    view(A, C, B) -> permute(0, 2, 1) -> softmax(-1) -> permute(0, 2, 1) -> view(original)

where C is the softmax axis, A is the product of the dimensions before it and B
the product of those after. `view`, `permute` and last-dimension `softmax` all
have XNNPACK partitioner configs, so the whole region rejoins the delegate.

Softmax normalises independently along one axis, so this computes exactly the
same function; in fp32 the two paths differ only by kernel rounding.

Why flatten to 3-D instead of permuting in place
------------------------------------------------
The obvious version of this repair for a 4-D tensor is
`x.permute(0, 2, 3, 1) -> softmax(-1) -> permute(0, 3, 1, 2)`. It is
mathematically identical, it fully delegates, and on ExecuTorch 1.4.0 it
SILENTLY PRODUCES WRONG RESULTS whenever the softmax input comes from a node
XNNPACK evaluates in NHWC layout (any convolution, or a bilinear resize).
XNNPACK's channels-last tagging does not account for an explicit permute that is
itself a layout change, so the axes get transposed twice. Measured on a
segmentation model: max absolute error 4.75e-02, and only 15.3% of pixels kept
the correct predicted class.

3-D tensors are not subject to that channels-last tagging, which is why the
flatten-first form is used here and the 4-D permute form is not. The
feasibility study documents the repro in detail.

Where this runs
---------------
On the ExportedProgram produced by `torch.export.export`, i.e. BEFORE
`to_edge_transform_and_lower`. That is the last stage where the graph is plain
ATen operators. A .pte file cannot be repaired directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.export import ExportedProgram

RULE_ID = "DD-001"
RULE_TITLE = "non-last-dimension softmax"

# torch.export leaves softmax as the composite `aten.softmax.int`; it is only
# decomposed into `aten._softmax.default` later, inside to_edge. Matching both
# means the rule works whichever stage it is handed.
SOFTMAX_TARGETS = (
    torch.ops.aten.softmax.int,
    torch.ops.aten._softmax.default,
)

# Portable kernel names that ETDump reports when this fallback is hit. Used to
# connect a measured runtime hotspot to this repair rule.
PORTABLE_KERNEL_NAMES = (
    "native_call__softmax.out",
    "native_call_softmax.out",
)


@dataclass
class Detection:
    """One place in the graph where DD-001 applies."""

    node_name: str
    input_shape: tuple
    softmax_dim: int      # already normalised to be non-negative
    tensor_rank: int

    @property
    def last_dim(self) -> int:
        return self.tensor_rank - 1

    @property
    def vector_length(self) -> int:
        """How many elements are normalised together."""
        return self.input_shape[self.softmax_dim]

    @property
    def vector_count(self) -> int:
        """How many independent softmax vectors this node computes."""
        return math.prod(self.input_shape) // self.vector_length

    @property
    def element_stride(self) -> int:
        """Distance in elements between two members of the same vector.

        A large stride is why the portable kernel is slow: consecutive reads
        land in different cache lines.
        """
        if self.softmax_dim + 1 >= self.tensor_rank:
            return 1
        return math.prod(self.input_shape[self.softmax_dim + 1:])

    def explain(self) -> str:
        return (
            f"{self.node_name}: softmax(dim={self.softmax_dim}) on "
            f"{list(self.input_shape)} · rank {self.tensor_rank} · "
            f"last dim {self.last_dim}\n"
            f"  access: {self.vector_count:,} vectors x {self.vector_length} "
            f"classes, stride {self.element_stride:,}"
        )


@dataclass
class SkipReason:
    """A softmax node that was examined but deliberately not repaired."""

    node_name: str
    reason: str


@dataclass
class DetectionResult:
    detections: List[Detection]
    skipped: List[SkipReason]

    @property
    def applies(self) -> bool:
        return len(self.detections) > 0


def _normalise_dim(dim: int, rank: int) -> int:
    """Convert a possibly-negative dim into a non-negative index."""
    if dim < 0:
        return dim + rank
    return dim


def _softmax_input_shape(node: torch.fx.Node) -> Optional[tuple]:
    """Static shape of a softmax node's input, or None if unavailable.

    `meta['val']` is the fake tensor torch.export recorded during tracing. If it
    is missing, or if any dimension is symbolic (a dynamic shape), we cannot
    compute the flattened sizes the rewrite needs.
    """
    input_node = node.args[0]
    fake_tensor = input_node.meta.get("val", None)
    if fake_tensor is None:
        return None

    shape = []
    for size in fake_tensor.shape:
        # Only a plain Python int is safe. A dynamic dimension arrives as a
        # torch.SymInt, and calling int() on one silently returns the example
        # value it was traced with - which would bake that single size into the
        # rewrite's view() and break every other batch size.
        if not isinstance(size, int):
            return None
        shape.append(size)
    return tuple(shape)


def detect(exported_program: ExportedProgram) -> DetectionResult:
    """Find every softmax in the graph that XNNPACK will refuse.

    Every softmax node is examined and put into exactly one bucket: repairable
    (a Detection) or deliberately skipped (a SkipReason with an explanation).
    """
    detections: List[Detection] = []
    skipped: List[SkipReason] = []

    for node in exported_program.graph.nodes:
        if node.op != "call_function" or node.target not in SOFTMAX_TARGETS:
            continue

        shape = _softmax_input_shape(node)
        if shape is None:
            skipped.append(SkipReason(
                node.name,
                "input shape is unavailable or dynamic, so the flattened sizes "
                "needed for the rewrite cannot be computed",
            ))
            continue

        rank = len(shape)
        if rank < 2:
            skipped.append(SkipReason(
                node.name,
                f"tensor rank is {rank}; the rewrite needs at least 2 dimensions",
            ))
            continue

        softmax_dim = _normalise_dim(int(node.args[1]), rank)
        if softmax_dim < 0 or softmax_dim >= rank:
            skipped.append(SkipReason(
                node.name,
                f"softmax dim {node.args[1]} is out of range for rank {rank}",
            ))
            continue

        if softmax_dim == rank - 1:
            skipped.append(SkipReason(
                node.name,
                "softmax is already on the last dimension, so XNNPACK accepts it",
            ))
            continue

        if 0 in shape:
            skipped.append(SkipReason(
                node.name,
                f"input shape {shape} has a zero-sized dimension",
            ))
            continue

        detections.append(Detection(
            node_name=node.name,
            input_shape=shape,
            softmax_dim=softmax_dim,
            tensor_rank=rank,
        ))

    return DetectionResult(detections=detections, skipped=skipped)


def apply(exported_program: ExportedProgram) -> int:
    """Rewrite every detected site, in place. Returns how many were repaired.

    The caller is expected to hand in a copy if it wants to keep the original
    (the DelegateDoctor pipeline does exactly that, so it can export both).
    """
    detection_result = detect(exported_program)
    repairable_node_names = {d.node_name for d in detection_result.detections}
    if not repairable_node_names:
        return 0

    graph = exported_program.graph
    repaired_count = 0

    # list(...) because we mutate the graph while walking it.
    for node in list(graph.nodes):
        if node.name not in repairable_node_names:
            continue

        input_node = node.args[0]
        input_fake_tensor = input_node.meta["val"]
        original_shape = [int(size) for size in input_fake_tensor.shape]
        rank = len(original_shape)
        softmax_dim = _normalise_dim(int(node.args[1]), rank)

        # Collapse to three dimensions around the softmax axis.
        #   before_size: everything to the left of the axis
        #   axis_size:   the axis being normalised
        #   after_size:  everything to the right of the axis
        before_size = math.prod(original_shape[:softmax_dim]) if softmax_dim > 0 else 1
        axis_size = original_shape[softmax_dim]
        after_size = (
            math.prod(original_shape[softmax_dim + 1:])
            if softmax_dim + 1 < rank
            else 1
        )

        # Any extra arguments the original op carried, e.g. `half_to_float` for
        # aten._softmax or `dtype` for aten.softmax.int. Passed through so the
        # rewrite cannot change the operator's behaviour.
        extra_args = tuple(node.args[2:])

        with graph.inserting_before(node):
            reshaped = graph.call_function(
                torch.ops.aten.view.default,
                (input_node, [before_size, axis_size, after_size]),
            )
            reshaped.meta["val"] = torch.ops.aten.view.default(
                input_fake_tensor, [before_size, axis_size, after_size]
            )

            # Softmax axis is now last.
            moved = graph.call_function(
                torch.ops.aten.permute.default, (reshaped, [0, 2, 1])
            )
            moved.meta["val"] = torch.ops.aten.permute.default(
                reshaped.meta["val"], [0, 2, 1]
            )

            softmaxed = graph.call_function(node.target, (moved, -1, *extra_args))
            softmaxed.meta["val"] = node.target(moved.meta["val"], -1, *extra_args)

            moved_back = graph.call_function(
                torch.ops.aten.permute.default, (softmaxed, [0, 2, 1])
            )
            moved_back.meta["val"] = torch.ops.aten.permute.default(
                softmaxed.meta["val"], [0, 2, 1]
            )

            restored = graph.call_function(
                torch.ops.aten.view.default, (moved_back, list(original_shape))
            )
            restored.meta["val"] = torch.ops.aten.view.default(
                moved_back.meta["val"], list(original_shape)
            )

        # Carry provenance across so ETRecord/Inspector can still attribute the
        # rewritten region back to the original source line.
        for new_node in (reshaped, moved, softmaxed, moved_back, restored):
            if "nn_module_stack" in node.meta:
                new_node.meta["nn_module_stack"] = node.meta["nn_module_stack"]
            if "stack_trace" in node.meta:
                new_node.meta["stack_trace"] = node.meta["stack_trace"]

        node.replace_all_uses_with(restored)
        graph.erase_node(node)
        repaired_count += 1

    if repaired_count > 0:
        graph.lint()
        exported_program.graph_module.recompile()
    return repaired_count


def describe_rewrite() -> str:
    """The before/after shown in the terminal report."""
    return "softmax(dim=D) -> view -> permute -> softmax(dim=-1) -> permute -> view"


def matches_portable_kernel(kernel_name: str) -> bool:
    """Does this ETDump portable-kernel name correspond to DD-001?

    Used to link a measured runtime hotspot to this repair rule.
    """
    return kernel_name in PORTABLE_KERNEL_NAMES
