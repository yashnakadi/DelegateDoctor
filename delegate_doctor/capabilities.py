"""What DelegateDoctor can currently push to, and read back from, the device.

Exporting a graph and running it on the Arm target are separate abilities, and
this module is where the second one is decided. It exists so that a model using
an int64 input or returning two tensors is *analyzed* rather than rejected:
`torch.export` decides whether the graph is capturable, and this decides how far
down the device pipeline that graph can then travel.

The limits below are the runner's, not PyTorch's. `executor_runner --inputs`
reads raw fp32 blobs with no header and no names, so today the transport carries
positional fp32 tensors and nothing else. Saying that precisely is the point.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# The runner reads and writes raw tensor data with no dtype tag, so the dtype
# has to be agreed in advance. fp32 is what both sides assume.
TRANSPORTABLE_DTYPE = torch.float32


@dataclass
class Capability:
    """Whether a stage can run, and - when it cannot - exactly why not."""

    supported: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.supported


SUPPORTED = Capability(True)


def assess_inputs(args: tuple, kwargs: dict) -> Capability:
    """Can these arguments be staged as files for the Android runner?"""
    if kwargs:
        names = ", ".join(sorted(kwargs))
        return Capability(False, (
            f"the Android input transport passes positional tensors only, and "
            f"this model was exported with keyword arguments ({names})"
        ))

    if not args:
        return Capability(False, "the model takes no positional tensor inputs")

    for position, value in enumerate(args):
        if not isinstance(value, torch.Tensor):
            return Capability(False, (
                f"input {position} is {type(value).__name__}; the Android input "
                f"transport carries tensors only"
            ))
        if value.dtype != TRANSPORTABLE_DTYPE:
            return Capability(False, (
                f"input {position} is {value.dtype}; the Android input transport "
                f"writes raw fp32 blobs with no dtype header"
            ))
        if not torch.isfinite(value).all():
            return Capability(False, (
                f"input {position} contains NaN or infinity, so a numerical "
                f"comparison would be meaningless"
            ))
    return SUPPORTED


def assess_output(output) -> Capability:
    """Can this model's output be verified against the device's?

    Device verification reads back one raw fp32 tensor, so a model returning
    something richer is analyzed and benchmarked in every other respect while
    verification honestly reports that it could not check the result.
    """
    if isinstance(output, torch.Tensor):
        first, extra = output, 0
    elif isinstance(output, (tuple, list)) and output:
        first, extra = output[0], len(output) - 1
    else:
        return Capability(False, (
            f"the model returns {type(output).__name__}; device verification "
            f"reads back a single tensor"
        ))

    if not isinstance(first, torch.Tensor):
        return Capability(False, (
            f"the first output is {type(first).__name__}; device verification "
            f"reads back a tensor"
        ))
    if first.dtype != TRANSPORTABLE_DTYPE:
        return Capability(False, (
            f"the first output is {first.dtype}; device verification reads back "
            f"raw fp32 bytes"
        ))
    if extra:
        return Capability(False, (
            f"the model returns {extra + 1} outputs; device verification "
            f"currently compares the first tensor only, so it cannot confirm "
            f"the rest are unchanged"
        ))
    return SUPPORTED


def first_output_tensor(output):
    """The tensor the host gates compare, or None if there is not one."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        candidate = output[0]
        if isinstance(candidate, torch.Tensor):
            return candidate
    return None


def describe_arguments(args: tuple, kwargs: dict) -> str:
    """A one-line summary for the report. Never prints tensor values."""
    parts = []
    for value in args:
        if isinstance(value, torch.Tensor):
            shape = ",".join(str(int(size)) for size in value.shape)
            dtype = str(value.dtype).replace("torch.", "")
            parts.append(f"[{shape}]:{dtype}")
        else:
            parts.append(type(value).__name__)
    for name in sorted(kwargs):
        value = kwargs[name]
        if isinstance(value, torch.Tensor):
            shape = ",".join(str(int(size)) for size in value.shape)
            dtype = str(value.dtype).replace("torch.", "")
            parts.append(f"{name}=[{shape}]:{dtype}")
        else:
            parts.append(f"{name}={type(value).__name__}")

    count = len(args) + len(kwargs)
    plural = "input" if count == 1 else "inputs"
    return f"{count} {plural} · " + " ".join(parts)
