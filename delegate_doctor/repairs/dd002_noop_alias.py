"""DD-002 - no-op alias nodes that fragment the XNNPACK graph.

The problem
-----------
`aten.alias` returns a tensor with exactly the same values, shape and dtype as
its input. It carries no computation at all. PyTorch emits one whenever a slice
happens to cover an entire dimension, for example timm's GhostNet:

    # timm/models/ghostnet.py, GhostModule.forward
    out = torch.cat([x1, x2], dim=1)
    return out[:, :self.out_chs, :, :]      # covers everything -> aten.alias

XNNPACK's partitioner has no config for `alias` or `alias_copy` (see the 53
configs under backends/xnnpack/partition/config/), so every one of them drops
out of the delegate. The kernel itself is nearly free - a memcpy - but each node
splits the graph, and XNNPACK must convert layouts at every blob boundary.

On GhostNet-100 that is 32 alias nodes producing **49 delegate blobs**. Measured
on a physical RMX2030, the alias kernels themselves cost 3.3 ms (0.4% of
runtime) while the boundary layout conversions cost **176 ms across 128 nodes**.
The fallback is cheap; the fragmentation it causes is not.

The repair
----------
Delete the node and forward its input. One operator becomes zero.

This is exact, not approximate: an alias is the identity on values, and
ExecuTorch's exported graphs are functionalised, so there is no mutation or
aliasing relationship that removing it could break. Measured host and device
error is 0.000e+00 - bit-identical, not merely within tolerance.

Compare with the LayerNorm candidate that was rejected earlier, which turned one
fused operator into nine unfused tensor passes and ran slower. DD-002 goes the
other way: 1 op -> 0 ops, no arithmetic added.

Where this runs
---------------
On the ExportedProgram from `torch.export.export`, before
`to_edge_transform_and_lower`, exactly like DD-001.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch.export import ExportedProgram

RULE_ID = "DD-002"
RULE_TITLE = "no-op alias"

# `aten.alias` is what torch.export emits; `alias_copy` is its functional form,
# which is what survives into the edge dialect. Match either.
ALIAS_TARGETS = (
    torch.ops.aten.alias.default,
    torch.ops.aten.alias_copy.default,
)

# The portable kernel ETDump reports when this fallback is hit, used to connect
# a measured hotspot to this rule.
PORTABLE_KERNEL_NAMES = (
    "native_call_alias_copy.out",
    "native_call_alias.out",
)


@dataclass
class Detection:
    """One no-op alias that can be deleted."""

    node_name: str
    shape: tuple
    dtype: str

    def explain(self) -> str:
        return (
            f"{self.node_name}: alias on {list(self.shape)} ({self.dtype}) "
            f"- identity, no computation"
        )


@dataclass
class SkipReason:
    node_name: str
    reason: str


@dataclass
class DetectionResult:
    detections: List[Detection]
    skipped: List[SkipReason]

    @property
    def applies(self) -> bool:
        return len(self.detections) > 0


def _static_shape(fake_tensor):
    """Return a concrete shape tuple, or None if any dimension is symbolic."""
    if fake_tensor is None:
        return None
    shape = []
    for size in fake_tensor.shape:
        # A dynamic dimension arrives as a torch.SymInt; int() on one silently
        # returns the traced example value, so only a real int is accepted.
        if not isinstance(size, int):
            return None
        shape.append(size)
    return tuple(shape)


def detect(exported_program: ExportedProgram) -> DetectionResult:
    """Find every alias node that is provably an identity.

    An alias is only removed when its input and output agree on shape and
    dtype. Anything else is left alone with a recorded reason rather than
    guessed at.
    """
    detections: List[Detection] = []
    skipped: List[SkipReason] = []

    for node in exported_program.graph.nodes:
        if node.op != "call_function" or node.target not in ALIAS_TARGETS:
            continue

        input_value = node.args[0].meta.get("val")
        output_value = node.meta.get("val")
        input_shape = _static_shape(input_value)
        output_shape = _static_shape(output_value)

        if input_shape is None or output_shape is None:
            skipped.append(SkipReason(
                node.name,
                "input or output shape is unavailable or dynamic",
            ))
            continue

        if input_shape != output_shape:
            skipped.append(SkipReason(
                node.name,
                f"shape changes {input_shape} -> {output_shape}, so this is not "
                f"an identity",
            ))
            continue

        if input_value.dtype != output_value.dtype:
            skipped.append(SkipReason(
                node.name,
                f"dtype changes {input_value.dtype} -> {output_value.dtype}, so "
                f"this is not an identity",
            ))
            continue

        detections.append(Detection(
            node_name=node.name,
            shape=output_shape,
            dtype=str(output_value.dtype).replace("torch.", ""),
        ))

    return DetectionResult(detections=detections, skipped=skipped)


def apply(exported_program: ExportedProgram) -> int:
    """Delete every detected alias, forwarding its input. Returns how many.

    The caller hands in a copy if it wants to keep the original; the
    DelegateDoctor pipeline does exactly that so it can export both.
    """
    removable = {d.node_name for d in detect(exported_program).detections}
    if not removable:
        return 0

    graph = exported_program.graph
    removed = 0
    # list(...) because the graph is mutated while walking it.
    for node in list(graph.nodes):
        if node.name not in removable:
            continue
        node.replace_all_uses_with(node.args[0])
        graph.erase_node(node)
        removed += 1

    if removed:
        graph.lint()
        exported_program.graph_module.recompile()
    return removed


def describe_rewrite() -> str:
    return "alias(x) -> x   (node deleted, 1 op -> 0 ops)"


def matches_portable_kernel(kernel_name: str) -> bool:
    """Does this ETDump portable-kernel name correspond to DD-002?"""
    return kernel_name in PORTABLE_KERNEL_NAMES
