"""Load a user's own PyTorch model from a Python file.

`delegate-doctor optimize my_model.py` runs DelegateDoctor against a model it
has never seen, without the user editing anything inside this package.

The contract is deliberately tiny - two functions, no configuration:

    def create_model():        # returns a torch.nn.Module
        ...

    def example_inputs():      # returns a tuple of positional tensors
        ...

SECURITY: this module imports and executes the supplied Python file. There is no
sandbox and none is attempted. Only point `optimize` at files you trust.

Everything is validated before any expensive export or device work happens, so a
contract mistake costs a second rather than several minutes.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import uuid

import torch

from .export_model import ModelSpec

# DelegateDoctor's export, verification and device paths are fp32-only today.
SUPPORTED_DTYPE = torch.float32

REQUIRED_FUNCTIONS = ("create_model", "example_inputs")


class CustomModelError(RuntimeError):
    """A problem with the user's model file, reported without a traceback."""


def load_module(path: str):
    """Import a .py file under a unique throwaway module name.

    A generated name keeps repeated runs, and the test suite, from colliding
    with each other or with anything already in sys.modules. The module is
    registered while executing (torch.export re-imports the defining module
    during tracing) and removed again afterwards.
    """
    resolved = os.path.abspath(os.path.expanduser(path))

    if not os.path.exists(resolved):
        raise CustomModelError(f"Model file not found: {path}")
    if not os.path.isfile(resolved):
        raise CustomModelError(f"Not a file: {path}")
    if not resolved.endswith(".py"):
        raise CustomModelError(
            f"Not a Python file: {path}\n"
            f"`optimize` expects a .py file defining create_model() and "
            f"example_inputs()."
        )

    module_name = f"delegate_doctor_custom_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise CustomModelError(f"Could not load {path} as a Python module.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    # The file's own directory goes on sys.path so `import my_project` works
    # from a model file that sits next to its project. Restored afterwards.
    directory = os.path.dirname(resolved)
    added_to_path = directory not in sys.path
    if added_to_path:
        sys.path.insert(0, directory)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(module_name, None)
        raise CustomModelError(
            f"Failed to import {path}\n\n{type(error).__name__}: {error}"
        )
    finally:
        if added_to_path and directory in sys.path:
            sys.path.remove(directory)

    return module, resolved


def check_contract(module, path: str) -> None:
    """Confirm the two required functions exist and are callable."""
    missing = [name for name in REQUIRED_FUNCTIONS if not hasattr(module, name)]
    if missing:
        listed = "\n".join(f"  {name}()" for name in missing)
        raise CustomModelError(
            f"Invalid custom model file: {path}\n"
            f"\n"
            f"Missing required function(s):\n{listed}\n"
            f"\n"
            f"A model file must define:\n"
            f"  def create_model():    # -> torch.nn.Module\n"
            f"  def example_inputs():  # -> tuple[torch.Tensor, ...]"
        )
    for name in REQUIRED_FUNCTIONS:
        if not callable(getattr(module, name)):
            raise CustomModelError(
                f"Invalid custom model file: {path}\n\n{name} is not callable."
            )


def build_model(module, path: str) -> torch.nn.Module:
    """Call create_model() and check what came back."""
    try:
        model = module.create_model()
    except Exception as error:
        raise CustomModelError(
            f"create_model() raised an exception in {path}\n\n"
            f"{type(error).__name__}: {error}"
        )

    if not isinstance(model, torch.nn.Module):
        raise CustomModelError(
            f"Invalid create_model()\n\n"
            f"Expected:\n  torch.nn.Module\n\nGot:\n  {type(model).__name__}"
        )

    # A forgotten .eval() is a mistake, not a reason to refuse the model.
    # Dropout and BatchNorm behave differently in training mode, so we switch
    # and say so rather than silently exporting a training-mode graph.
    switched = model.training
    model.eval()
    if switched:
        print("Note: model was in training mode; switched to eval() for export.")
    return model


def build_inputs(module, path: str) -> tuple:
    """Call example_inputs() exactly once and validate the tensors.

    Called once on purpose: example_inputs() may use torch.randn, and every
    stage of the run - export, host verification, device verification and the
    benchmark - must see identical values.
    """
    try:
        inputs = module.example_inputs()
    except Exception as error:
        raise CustomModelError(
            f"example_inputs() raised an exception in {path}\n\n"
            f"{type(error).__name__}: {error}"
        )

    if not isinstance(inputs, tuple):
        raise CustomModelError(
            f"Invalid example_inputs()\n\n"
            f"Expected:\n  tuple[torch.Tensor, ...]\n\n"
            f"Got:\n  {type(inputs).__name__}"
            + ("\n\nDid you mean `return (tensor,)`?"
               if isinstance(inputs, torch.Tensor) else "")
        )
    if len(inputs) == 0:
        raise CustomModelError(
            "Invalid example_inputs()\n\nThe tuple is empty; at least one input "
            "tensor is required."
        )

    for position, value in enumerate(inputs):
        if not isinstance(value, torch.Tensor):
            raise CustomModelError(
                f"Invalid example_inputs()\n\n"
                f"Input {position} is {type(value).__name__}, expected "
                f"torch.Tensor.\n"
                f"`optimize` currently supports positional tensor inputs only."
            )
        if value.dtype != SUPPORTED_DTYPE:
            raise CustomModelError(
                f"Unsupported input dtype: {value.dtype}\n"
                f"DelegateDoctor currently supports fp32 custom-model inputs.\n"
                f"(input {position}, shape {list(value.shape)})"
            )
        if not torch.isfinite(value).all():
            raise CustomModelError(
                f"Input {position} contains NaN or infinity. Verification "
                f"compares outputs numerically, so the input must be finite."
            )
    return inputs


def check_output(model: torch.nn.Module, inputs: tuple, path: str) -> None:
    """Run the model once to confirm the output shape DelegateDoctor can verify.

    Device verification reads back the first output tensor as raw fp32 bytes, so
    the model must produce a tensor (or a tuple/list whose first element is
    one). Checking here costs a single forward pass and saves a full export plus
    several minutes of device work when the answer is no.
    """
    try:
        with torch.no_grad():
            output = model(*inputs)
    except Exception as error:
        raise CustomModelError(
            f"The model raised an exception on the example inputs ({path})\n\n"
            f"{type(error).__name__}: {error}"
        )

    first = output[0] if isinstance(output, (tuple, list)) and output else output
    if not isinstance(first, torch.Tensor):
        raise CustomModelError(
            f"Unsupported model output.\n\n"
            f"DelegateDoctor custom-model verification currently requires a "
            f"single fp32 tensor output (or a tuple whose first element is "
            f"one).\nGot: {type(output).__name__}"
        )
    if first.dtype != SUPPORTED_DTYPE:
        raise CustomModelError(
            f"Unsupported model output dtype: {first.dtype}\n"
            f"DelegateDoctor custom-model verification currently requires fp32."
        )


def describe_inputs(inputs: tuple) -> str:
    """A one-line summary for the report. Never prints tensor values."""
    shapes = " ".join("[" + ",".join(str(int(s)) for s in t.shape) + "]" for t in inputs)
    plural = "tensor" if len(inputs) == 1 else "tensors"
    return f"{len(inputs)} {plural} · fp32 · {shapes}"


def load(path: str) -> ModelSpec:
    """Load, validate and wrap a user's model file as a ModelSpec.

    The returned spec goes through exactly the same pipeline as the built-in
    examples.
    """
    module, resolved = load_module(path)
    check_contract(module, path)
    model = build_model(module, path)
    inputs = build_inputs(module, path)
    check_output(model, inputs, path)

    # The module deliberately stays in sys.modules: torch.export re-imports the
    # module that defines the model's class while tracing, and export happens
    # later in the pipeline. The generated UUID name keeps repeated runs from
    # colliding with each other or with anything the user has imported.

    # Filename-derived name; a module-level MODEL_NAME wins if the user set one.
    name = getattr(module, "MODEL_NAME", None) or os.path.basename(resolved)

    return ModelSpec(
        name=str(name),
        model=model,
        example_inputs=inputs,
        # Custom models have unknown output semantics, so no argmax claim is
        # made. Verification reports tensor error only. See README.
        argmax_dim=None,
        description=f"custom model from {resolved}",
    )
