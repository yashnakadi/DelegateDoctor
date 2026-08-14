"""Load a `.pt2` exported program and the inputs used to exercise it.

NOT a public input path. `.pt2` and `.pt` were once accepted by the CLI; they
are now internal artifacts, and `model_source` recognises those suffixes only
in order to refuse them with an explanation. What remains here is the loader
itself, used to build a `ModelSpec` from a serialized program - in practice by
the test suite, which needs a program without going through a source file.

    model.pt2   a serialized torch.export.ExportedProgram
    inputs.pt   the positional arguments that program is called with

    torch.export.save(exported_program, "model.pt2")
    torch.save(example_inputs, "inputs.pt")

Starting from PyTorch's own exported graph means DelegateDoctor never has to
understand anybody's Python. The graph in the `.pt2` is the source of truth: it
is what gets lowered, analyzed, repaired and verified.

    .pt2 = PyTorch ExportedProgram, the input to DelegateDoctor
    .pte = ExecuTorch deployment artifact, the output

Everything here runs before any expensive work, so a mistake costs a second
rather than several minutes of device time.

SECURITY: `inputs.pt` is read with `torch.load(..., weights_only=True)`, which
restricts unpickling to tensors and plain containers. Input artifacts holding
custom classes are rejected rather than unpickled - there is no fallback to
unrestricted loading. `torch.export.load` is PyTorch's supported deserializer
for `.pt2` and reads the program's constants through the same restricted path,
but it is still a deserializer: load `.pt2` files from sources you trust.
"""

from __future__ import annotations

import os
import re

import torch

MODEL_SUFFIX = ".pt2"
INPUTS_SUFFIX = ".pt"

# DelegateDoctor's export, verification and device paths are fp32-only today.
SUPPORTED_DTYPE = torch.float32

_URL_SCHEMES = ("http://", "https://", "git://", "git@", "ssh://", "ftp://",
                "file://")
_BARE_HOST = re.compile(r"^(www\.)?[\w-]+(\.[\w-]+)+/")

EXPORT_HINT = (
    "Create the two files with PyTorch:\n"
    "\n"
    "  exported = torch.export.export(model.eval(), example_inputs)\n"
    "  torch.export.save(exported, \"model.pt2\")\n"
    "  torch.save(example_inputs, \"inputs.pt\")"
)


class ModelInputError(RuntimeError):
    """The supplied model or input artifact cannot be used, with the reason."""


def looks_like_url(target: str) -> bool:
    """Is this a remote address rather than a local path?

    Used only to produce a clear "not supported" error. DelegateDoctor has no
    remote-fetch code path for it to fall back to.
    """
    text = (target or "").strip()
    if text.lower().startswith(_URL_SCHEMES):
        return True
    return bool(_BARE_HOST.match(text)) and not os.path.exists(text)


UNSUPPORTED_INPUT_MESSAGE = (
    "unsupported model input: DelegateDoctor expects a local .pt2 "
    "ExportedProgram\n"
    "\n"
    "  delegate-doctor optimize model.pt2 --inputs inputs.pt\n"
    "\n"
    "Remote sources - repository URLs, any http(s) address - are not supported.\n"
    "\n" + EXPORT_HINT
)


def _resolve_file(target: str, suffix: str, label: str, extra: str = "") -> str:
    """Shared checks: not a URL, exists, regular file, right suffix."""
    if not target or not str(target).strip():
        raise ModelInputError(f"No {label} given.\n\n{UNSUPPORTED_INPUT_MESSAGE}")

    if looks_like_url(target):
        raise ModelInputError(UNSUPPORTED_INPUT_MESSAGE)

    path = os.path.abspath(os.path.expanduser(str(target)))

    if os.path.isdir(path):
        raise ModelInputError(
            f"{label} is a directory: {target}\n"
            f"\nDelegateDoctor takes a single {suffix} file."
        )
    if not os.path.exists(path):
        raise ModelInputError(
            f"{label} not found: {target}\n\n{extra or EXPORT_HINT}"
        )
    # isfile() is a regular-file test, so device nodes, FIFOs and sockets are
    # rejected here rather than blocking forever inside a read.
    if not os.path.isfile(path):
        raise ModelInputError(
            f"{label} is not a regular file: {target}\n"
            f"\nDevices, pipes and sockets are not supported."
        )
    if not path.endswith(suffix):
        raise ModelInputError(
            f"{label} must be a {suffix} file: {target}{extra}"
        )
    return path


def resolve_model_path(target: str) -> str:
    """Validate the `.pt2` path without opening it yet."""
    if target and str(target).endswith(".pte"):
        raise ModelInputError(
            f"That is an ExecuTorch artifact, not an exported program: {target}\n"
            f"\n"
            f"  .pt2 = PyTorch ExportedProgram, the input to DelegateDoctor\n"
            f"  .pte = ExecuTorch deployment artifact, the output\n"
            f"\n"
            f"DelegateDoctor cannot repair a .pte: its delegated regions are\n"
            f"already compiled blobs. Re-export the model instead.\n"
            f"\n{EXPORT_HINT}"
        )
    if target and str(target).endswith(".py"):
        raise ModelInputError(
            f"Python source is no longer a DelegateDoctor input: {target}\n"
            f"\n"
            f"Export the model first, then optimize the exported program:\n"
            f"\n{EXPORT_HINT}"
        )
    return _resolve_file(target, MODEL_SUFFIX, "Model file")


def resolve_inputs_path(target: str) -> str:
    """Validate the `inputs.pt` path without opening it yet."""
    return _resolve_file(
        target, INPUTS_SUFFIX, "Inputs file",
        extra=(
            "\n\nThe inputs file holds the positional arguments the exported "
            "program is called with:\n"
            "\n  torch.save((torch.randn(1, 3, 224, 224),), \"inputs.pt\")"
        ),
    )


def load_exported_program(path: str) -> torch.export.ExportedProgram:
    """Deserialize the `.pt2` with PyTorch's supported loader.

    No custom pickle handling: whatever `torch.export.load` accepts is what
    DelegateDoctor accepts.
    """
    try:
        loaded = torch.export.load(path)
    except Exception as error:
        raise ModelInputError(
            f"Could not load the exported program: {os.path.basename(path)}\n"
            f"\n"
            f"A .pt2 must be written by torch.export.save().\n"
            f"\n"
            f"{type(error).__name__}: {str(error)[:600]}\n"
            f"\n{EXPORT_HINT}"
        )

    if not isinstance(loaded, torch.export.ExportedProgram):
        raise ModelInputError(
            f"{os.path.basename(path)} did not contain an ExportedProgram.\n"
            f"\nGot: {type(loaded).__name__}\n\n{EXPORT_HINT}"
        )

    try:
        module = loaded.module()
    except Exception as error:
        raise ModelInputError(
            f"The exported program in {os.path.basename(path)} could not be "
            f"turned into a callable module.\n"
            f"\n{type(error).__name__}: {str(error)[:600]}"
        )
    if not callable(module):
        raise ModelInputError(
            f"The exported program in {os.path.basename(path)} is not callable."
        )
    return loaded


def load_inputs(path: str) -> tuple:
    """Load and validate the representative inputs.

    Only tensors and plain tuples/lists are supported, which is exactly what
    `weights_only=True` will unpickle. A custom object in the artifact is
    rejected rather than being executed during deserialization.
    """
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ModelInputError(
            f"Could not load the inputs file: {os.path.basename(path)}\n"
            f"\n"
            f"DelegateDoctor loads inputs with weights_only=True, so the file "
            f"may contain\nonly tensors and plain tuples or lists. Custom "
            f"objects are rejected rather than\nunpickled.\n"
            f"\n"
            f"{type(error).__name__}: {str(error)[:600]}\n"
            f"\n"
            f"  torch.save((torch.randn(1, 3, 224, 224),), \"inputs.pt\")"
        )
    return normalize_inputs(loaded, path)


def normalize_inputs(loaded, path: str) -> tuple:
    """Accept a tensor or a tuple/list of tensors; reject anything else clearly."""
    name = os.path.basename(path)

    if isinstance(loaded, torch.Tensor):
        # A single tensor is unambiguous, so wrap it rather than nag.
        inputs = (loaded,)
    elif isinstance(loaded, (tuple, list)):
        inputs = tuple(loaded)
    else:
        raise ModelInputError(
            f"Unsupported input structure in {name}: {type(loaded).__name__}\n"
            f"\n"
            f"DelegateDoctor currently supports a tuple of positional tensors.\n"
            f"Keyword arguments and nested containers are not supported.\n"
            f"\n"
            f"  torch.save((torch.randn(1, 3, 224, 224),), \"inputs.pt\")"
        )

    if not inputs:
        raise ModelInputError(
            f"{name} contains no inputs.\n"
            f"\nAt least one input tensor is required."
        )

    for position, value in enumerate(inputs):
        if not isinstance(value, torch.Tensor):
            raise ModelInputError(
                f"Unsupported input structure in {name}\n"
                f"\n"
                f"Input {position} is {type(value).__name__}, expected "
                f"torch.Tensor.\n"
                f"DelegateDoctor currently supports positional tensor inputs only."
            )
        if value.dtype != SUPPORTED_DTYPE:
            raise ModelInputError(
                f"Unsupported input dtype in {name}: {value.dtype}\n"
                f"DelegateDoctor currently supports fp32 inputs.\n"
                f"(input {position}, shape {list(value.shape)})"
            )
        if not torch.isfinite(value).all():
            raise ModelInputError(
                f"Input {position} in {name} contains NaN or infinity.\n"
                f"Verification compares outputs numerically, so inputs must be "
                f"finite."
            )
    return inputs


def check_executes(exported_program, inputs: tuple, model_name: str):
    """Run the exported program once, before any repair or device work.

    Confirms the inputs actually fit the graph, and that the output is
    something DelegateDoctor can verify: device verification reads back the
    first output tensor as raw fp32 bytes.
    """
    try:
        with torch.no_grad():
            output = exported_program.module()(*inputs)
    except Exception as error:
        shapes = ", ".join(str(list(tensor.shape)) for tensor in inputs)
        raise ModelInputError(
            f"The inputs are not compatible with the exported program.\n"
            f"\n"
            f"  Model:  {model_name}\n"
            f"  Inputs: {len(inputs)} tensor(s) - {shapes}\n"
            f"\n"
            f"{type(error).__name__}: {str(error)[:600]}\n"
            f"\n"
            f"The inputs file must hold the same positional arguments the model "
            f"was\nexported with."
        )

    first = output[0] if isinstance(output, (tuple, list)) and output else output
    if not isinstance(first, torch.Tensor):
        raise ModelInputError(
            f"Unsupported model output: {type(output).__name__}\n"
            f"\n"
            f"DelegateDoctor verification currently requires a single fp32 "
            f"tensor output\n(or a tuple whose first element is one)."
        )
    if first.dtype != SUPPORTED_DTYPE:
        raise ModelInputError(
            f"Unsupported model output dtype: {first.dtype}\n"
            f"DelegateDoctor verification currently requires fp32."
        )
    return first


def describe_inputs(inputs: tuple) -> str:
    """A one-line summary for the report. Never prints tensor values."""
    shapes = " ".join("[" + ",".join(str(int(s)) for s in t.shape) + "]" for t in inputs)
    plural = "tensor" if len(inputs) == 1 else "tensors"
    return f"{len(inputs)} {plural} · fp32 · {shapes}"


def load_model_spec(model_target: str, inputs_target: str):
    """Validate both artifacts and wrap them as the ModelSpec the pipeline takes.

    Everything downstream sees an ExportedProgram, exactly as if it had come
    from a built-in demo model.
    """
    from .export_model import ModelSpec

    model_path = resolve_model_path(model_target)
    inputs_path = resolve_inputs_path(inputs_target)

    exported_program = load_exported_program(model_path)
    inputs = load_inputs(inputs_path)

    name = os.path.basename(model_path)
    check_executes(exported_program, inputs, name)

    return ModelSpec(
        name=name,
        exported_program=exported_program,
        example_args=inputs,
        # `inputs.pt` is a positional tuple by construction. The live Python API
        # accepts kwargs because it holds the real objects; this artifact path
        # deliberately stays narrower - see the module docstring.
        example_kwargs={},
        # Output semantics are unknown for a user's exported program, so no
        # argmax claim is made. Verification reports tensor error only.
        argmax_dim=None,
        description=f"exported program from {model_path}",
    )
