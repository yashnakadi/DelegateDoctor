"""DD-003 - `avg_pool2d` with `count_include_pad=True`, which XNNPACK refuses.

The problem
-----------
This is not a guess about why the operator falls back. ExecuTorch 1.4.0 states
the reason in its own partitioner, in `AvgPoolingConfig.check_constraints`
(backends/xnnpack/partition/config/generic_node_configs.py):

    if count_include_pad:
        why(node, reason="zero-padding in the averaging calculation is not "
                         "supported")
        return False

The check never looks at `padding`. Any `avg_pool2d` carrying
`count_include_pad=True` is rejected - and `True` is the ATen default, so a
model that simply never mentioned the argument gets rejected for a property its
author never chose.

torchvision's Inception V3 does exactly that. Every `BasicConv2d` branch calls
`F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)`, nine times, and all nine
land in the portable kernel. Measured on a physical Arm64 device that one
operator was ~60% of total model runtime and ~99.9% of all fallback runtime,
while the model was still 95.4% delegated *by operator count* - the gap this
project exists to expose.

The repair
----------
Materialise the padding as its own node, then tell the pooling operator that
there is no padding left to argue about:

    avg_pool2d(x, k, s, padding=p, count_include_pad=True)
      ->
    avg_pool2d(constant_pad_nd(x, [pw, pw, ph, ph], 0.0), k, s,
               padding=0, count_include_pad=False)

Why this is exact rather than close
-----------------------------------
`count_include_pad` decides one thing only: whether padded positions count
towards the divisor. With `ceil_mode=False`, every window lies wholly inside the
padded region, so the original divides by the full pooling region `k*k` for
every window. After the rewrite the padding is real data - explicit zeros - so
`count_include_pad=False` also divides by `k*k`, because there is no padding
left for it to exclude. The numerator is unchanged either way: a padded zero
contributes 0.0 to the sum in both forms.

So the two forms do not merely agree to a tolerance. Measured across shapes,
kernels, strides and paddings, the maximum absolute difference is 0.000e+00 -
the same bit-exactness DD-002 has.

`ceil_mode=False` is what makes this true, and is enforced, not assumed. With
`ceil_mode=True` a window may extend past the padded edge; ATen then shrinks the
divisor to the part that fits, which pre-padding cannot reproduce. Such nodes
are skipped - and they would not delegate anyway, since the same partitioner
config rejects `ceil_mode` separately.

Why adding a node makes things faster
-------------------------------------
`constant_pad_nd` is itself delegable: `ConstantPadConfig` in the same file
rejects only *negative* padding. So one portable operator becomes two delegated
ones, and the delegate boundary the fallback used to force disappears with it.
This is the opposite trade from the LayerNorm candidate that was rejected
earlier, which replaced one fused operator with nine unfused passes.

The degenerate case `padding=[0, 0]` needs no pad node at all - with nothing
padded, `count_include_pad` is already a no-op, and flipping the flag alone
makes the node delegable.

Where this runs
---------------
On the ExportedProgram from `torch.export.export`, before
`to_edge_transform_and_lower`, exactly like DD-001 and DD-002.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.export import ExportedProgram

RULE_ID = "DD-003"
RULE_TITLE = "avg_pool2d padding"

# Presentation metadata only: the node sequence the HTML report draws as a
# before/after diagram. Nothing reads these during detection or rewriting.
FLOW_BEFORE = ("producer", "avg_pool2d (portable)", "consumer")
FLOW_AFTER = ("producer", "constant_pad_nd", "avg_pool2d", "consumer")

AVGPOOL_TARGETS = (torch.ops.aten.avg_pool2d.default,)

# The portable kernel ETDump reports when this fallback is hit, used to connect
# a measured hotspot to this rule.
PORTABLE_KERNEL_NAMES = (
    "native_call_avg_pool2d.out",
)

# Positional layout of aten.avg_pool2d.default, past the input tensor.
# torch.export omits trailing arguments left at their default, so a node with
# `count_include_pad=True` and `ceil_mode=False` may carry as few as four args.
# Reading `node.args` alone would report the two arguments that decide whether
# this rewrite is legal as "absent" rather than as their real values, so the
# defaults below are applied explicitly.
_KERNEL_SIZE, _STRIDE, _PADDING, _CEIL_MODE, _COUNT_INCLUDE_PAD, _DIVISOR = range(1, 7)
_DEFAULTS = {
    _STRIDE: [],
    _PADDING: [0, 0],
    _CEIL_MODE: False,
    _COUNT_INCLUDE_PAD: True,
    _DIVISOR: None,
}


@dataclass
class Detection:
    """One `avg_pool2d` that can be re-expressed with explicit padding."""

    node_name: str
    shape: tuple
    dtype: str
    kernel_size: tuple
    stride: tuple
    padding: tuple

    @property
    def needs_pad_node(self) -> bool:
        return any(self.padding)

    def explain(self) -> str:
        detail = (
            f"kernel {list(self.kernel_size)}, stride {list(self.stride)}, "
            f"padding {list(self.padding)}"
        )
        if not self.needs_pad_node:
            detail += " - flag only, no padding to materialise"
        return (
            f"{self.node_name}: avg_pool2d on {list(self.shape)} ({self.dtype}) "
            f"- {detail}"
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


def _static_shape(fake_tensor) -> Optional[tuple]:
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


def _argument(node, index):
    value = node.args[index] if len(node.args) > index else _DEFAULTS[index]
    # Keyword form is legal for every argument past the input tensor.
    for name, position in (("stride", _STRIDE), ("padding", _PADDING),
                           ("ceil_mode", _CEIL_MODE),
                           ("count_include_pad", _COUNT_INCLUDE_PAD),
                           ("divisor_override", _DIVISOR)):
        if position == index and name in node.kwargs:
            return node.kwargs[name]
    return value


def _pair(value) -> tuple:
    """ATen accepts a scalar or a 1-element list where a 2-tuple is meant."""
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return (int(value[0]), int(value[0]))
        return tuple(int(v) for v in value)
    return (int(value), int(value))


def detect(exported_program: ExportedProgram) -> DetectionResult:
    """Find every `avg_pool2d` whose padding can be made explicit.

    A node qualifies only when the rewrite is provably value-identical:
    `count_include_pad` is True (otherwise there is nothing to fix and the node
    already delegates), `ceil_mode` is False, `divisor_override` is unset, and
    the shape is static. Everything else is left alone with a recorded reason.
    """
    detections: List[Detection] = []
    skipped: List[SkipReason] = []

    for node in exported_program.graph.nodes:
        if node.op != "call_function" or node.target not in AVGPOOL_TARGETS:
            continue

        count_include_pad = _argument(node, _COUNT_INCLUDE_PAD)
        ceil_mode = _argument(node, _CEIL_MODE)
        divisor_override = _argument(node, _DIVISOR)

        if not count_include_pad:
            skipped.append(SkipReason(
                node.name,
                "count_include_pad is already False, so XNNPACK accepts this "
                "node unchanged",
            ))
            continue

        if ceil_mode:
            skipped.append(SkipReason(
                node.name,
                "ceil_mode is True, so a window may extend past the padded edge "
                "and the divisor is not the full pooling region",
            ))
            continue

        if divisor_override is not None:
            skipped.append(SkipReason(
                node.name,
                f"divisor_override is {divisor_override}, which overrides the "
                f"divisor this rewrite relies on",
            ))
            continue

        input_value = node.args[0].meta.get("val")
        output_value = node.meta.get("val")
        input_shape = _static_shape(input_value)
        if input_shape is None or _static_shape(output_value) is None:
            skipped.append(SkipReason(
                node.name,
                "input or output shape is unavailable or dynamic",
            ))
            continue

        if len(input_shape) < 3:
            skipped.append(SkipReason(
                node.name,
                f"input rank is {len(input_shape)}, which is not a pooling input",
            ))
            continue

        padding = _pair(_argument(node, _PADDING))
        if any(p < 0 for p in padding):
            # constant_pad_nd would treat a negative value as a crop, and
            # XNNPACK's ConstantPadConfig rejects it outright.
            skipped.append(SkipReason(
                node.name,
                f"padding {list(padding)} is negative",
            ))
            continue

        kernel_size = _pair(_argument(node, _KERNEL_SIZE))
        stride = _argument(node, _STRIDE)
        # An empty stride means "same as kernel_size" in ATen.
        stride = kernel_size if not stride else _pair(stride)

        detections.append(Detection(
            node_name=node.name,
            shape=input_shape,
            dtype=str(input_value.dtype).replace("torch.", ""),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ))

    return DetectionResult(detections=detections, skipped=skipped)


def _rewrite_arguments(node, kernel_size, stride):
    """Rebuild the node's arguments in full positional form.

    Every argument is written out rather than relying on defaults, so the
    partitioner reads the same values this rule reasoned about: padding zero,
    `count_include_pad` False, `ceil_mode` False, no divisor override.
    """
    node.args = (node.args[0], list(kernel_size), list(stride), [0, 0],
                 False, False)
    node.kwargs = {}


def apply(exported_program: ExportedProgram) -> int:
    """Rewrite every detected pooling node. Returns how many were changed.

    The caller hands in a copy if it wants to keep the original; the
    DelegateDoctor pipeline does exactly that so it can export both.
    """
    result = detect(exported_program)
    if not result.detections:
        return 0

    by_name = {d.node_name: d for d in result.detections}
    graph = exported_program.graph
    changed = 0

    for node in list(graph.nodes):
        detection = by_name.get(node.name)
        if detection is None:
            continue

        if detection.needs_pad_node:
            source = node.args[0]
            height, width = detection.padding
            # constant_pad_nd counts from the last dimension backwards, so the
            # width pair comes first.
            pad = [width, width, height, height]
            with graph.inserting_before(node):
                padded = graph.call_function(
                    torch.ops.aten.constant_pad_nd.default,
                    (source, pad, 0.0),
                )
            input_value = source.meta.get("val")
            # Shape metadata is required downstream; derive it by running the
            # op on the input's own fake tensor rather than recomputing sizes.
            with input_value.fake_mode:
                padded.meta["val"] = torch.ops.aten.constant_pad_nd.default(
                    input_value, pad, 0.0)
            node.args = (padded,) + tuple(node.args[1:])

        # The output shape is unchanged - moving the padding out of the
        # operator does not move its boundaries - so node.meta["val"] stands.
        _rewrite_arguments(node, detection.kernel_size, detection.stride)
        changed += 1

    if changed:
        graph.lint()
        exported_program.graph_module.recompile()
    return changed


def describe_rewrite() -> str:
    return (
        "avg_pool2d(x, k, s, padding=p, count_include_pad=True) -> "
        "avg_pool2d(constant_pad_nd(x, p, 0.0), k, s, padding=0, "
        "count_include_pad=False)"
    )


def matches_portable_kernel(kernel_name: str) -> bool:
    """Does this ETDump portable-kernel name correspond to DD-003?

    Used to link a measured runtime hotspot to this repair rule.
    """
    return kernel_name in PORTABLE_KERNEL_NAMES
