"""The public Python API.

    from delegate_doctor import optimize

    result = optimize(model, args=(example_input,))

DelegateDoctor receives the actual model object you already have in memory -
your architecture, your configuration, your trained weights. It does not need to
know how the model was built, how the checkpoint was loaded, or how your project
is laid out, because `torch.export` answers all of that by capturing the graph.

Where the boundary is
---------------------
`torch.export.export()` decides what DelegateDoctor accepts. If PyTorch can
capture your model, DelegateDoctor analyzes the resulting graph as far as
ExecuTorch, XNNPACK and the attached Arm target allow - and says exactly where
it stopped. Only export failure is a rejected input, because then there is no
graph to analyze at all.

Your model object
-----------------
Exporting needs inference mode, but that is *your* object, so this module puts
it back: training mode is restored afterwards, parameters and buffers are never
written to, and nothing is moved between devices behind your back. A
DataParallel wrapper is read through rather than unwrapped in place.
"""

from __future__ import annotations

import contextlib
from typing import Optional

import torch

from . import pipeline
from .result import OptimizationResult

# Wrappers that hold the real model in `.module` and add nothing DelegateDoctor
# can export. Reading the attribute does not modify the caller's object.
_WRAPPERS = ("DataParallel", "DistributedDataParallel")


class ExportFailed(RuntimeError):
    """`torch.export.export()` could not capture the model.

    The only true unsupported-input failure: with no ExportedProgram there is
    nothing for DelegateDoctor to analyze.
    """


@contextlib.contextmanager
def _inference_mode(model: torch.nn.Module):
    """Put the model in eval mode for the export, then put it back.

    `torch.export` traces whichever mode the module is in, and a graph captured
    in training mode has dropout and batch-norm behaviour nobody wants to
    deploy. Restoring afterwards means calling `optimize()` in the middle of a
    training script does not quietly change what that script does next.
    """
    was_training = model.training
    try:
        model.eval()
        yield model
    finally:
        if was_training:
            model.train()


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """See through DataParallel-style wrappers without mutating the caller's object."""
    while type(model).__name__ in _WRAPPERS:
        inner = getattr(model, "module", None)
        if not isinstance(inner, torch.nn.Module):
            raise ExportFailed(
                f"PYTORCH EXPORT FAILED\n"
                f"\n"
                f"{type(model).__name__} does not hold an nn.Module in `.module`, "
                f"so DelegateDoctor cannot reach the model to export it.\n"
                f"\n"
                f"Pass the underlying module directly."
            )
        model = inner
    return model


def _check_on_cpu(model: torch.nn.Module) -> None:
    """Refuse politely rather than moving the caller's model between devices.

    The ExecuTorch/XNNPACK path this tool analyzes is CPU-only, and calling
    `.cpu()` on someone's live training model is not ours to do.
    """
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.device.type != "cpu":
            raise ExportFailed(
                f"PYTORCH EXPORT FAILED\n"
                f"\n"
                f"This model's parameters are on {tensor.device} (for example "
                f"{name}).\nDelegateDoctor analyzes the ExecuTorch/XNNPACK CPU "
                f"deployment path.\n"
                f"\n"
                f"DelegateDoctor will not move your model for you. Pass a CPU "
                f"copy:\n"
                f"\n"
                f"  import copy\n"
                f"  optimize(copy.deepcopy(model).cpu(), args=args)"
            )


def export_for_analysis(
    model: torch.nn.Module,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    *,
    dynamic_shapes=None,
    strict: bool = True,
) -> torch.export.ExportedProgram:
    """Capture the caller's model with `torch.export`, using only what they gave.

    No shape inference, no invented tensors, no source inspection: `args`,
    `kwargs` and `dynamic_shapes` go straight to PyTorch.
    """
    if not isinstance(model, torch.nn.Module):
        raise ExportFailed(
            f"PYTORCH EXPORT FAILED\n"
            f"\n"
            f"optimize() takes a torch.nn.Module; got {type(model).__name__}.\n"
            f"\n"
            f"  from delegate_doctor import optimize\n"
            f"  optimize(model, args=(example_input,))"
        )

    exportable = _unwrap(model)
    _check_on_cpu(exportable)

    with _inference_mode(exportable):
        try:
            return torch.export.export(
                exportable,
                args=tuple(args),
                kwargs=dict(kwargs or {}),
                dynamic_shapes=dynamic_shapes,
                strict=strict,
            )
        except Exception as error:
            raise ExportFailed(
                f"PYTORCH EXPORT FAILED\n"
                f"\n"
                f"DelegateDoctor could not capture this model as a torch.export "
                f"ExportedProgram.\n"
                f"The model has not entered the DelegateDoctor analysis "
                f"pipeline.\n"
                f"\n"
                f"PyTorch error:\n"
                f"{type(error).__name__}: {str(error)[:1200]}"
            ) from error


def analyze_exported_program(
    exported_program: torch.export.ExportedProgram,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    *,
    name: str = "exported program",
    description: str = "",
    argmax_dim: Optional[int] = None,
    **pipeline_options,
) -> OptimizationResult:
    """Run the pipeline on an `ExportedProgram` you already have.

    The single internal entry point: `optimize()`, the `.pt2` CLI and the demo
    catalog all arrive here. Use it directly when you exported the graph
    yourself and want DelegateDoctor to take it from there.
    """
    from .export_model import ModelSpec

    if not isinstance(exported_program, torch.export.ExportedProgram):
        raise ExportFailed(
            f"analyze_exported_program() takes a torch.export.ExportedProgram; "
            f"got {type(exported_program).__name__}."
        )

    spec = ModelSpec(
        name=name,
        exported_program=exported_program,
        example_args=tuple(args),
        example_kwargs=dict(kwargs or {}),
        argmax_dim=argmax_dim,
        description=description,
    )
    return pipeline.run_optimization(spec, **pipeline_options)


def optimize(
    model: torch.nn.Module,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    *,
    dynamic_shapes=None,
    strict: bool = True,
    name: Optional[str] = None,
    argmax_dim: Optional[int] = None,
    **pipeline_options,
) -> OptimizationResult:
    """Analyze a live PyTorch model, and repair it if a known pattern fits.

        result = optimize(model, args=(x,))
        result = optimize(model, args=(input_ids,),
                          kwargs={"attention_mask": mask})

    Arguments mirror `torch.export.export`. `dynamic_shapes` is forwarded
    unchanged; a dynamic graph is still analyzed, though a repair rule may
    decline to rewrite it.

    Returns an `OptimizationResult`. Analysis stopping early - no device
    attached, no matching repair, ExecuTorch declining the graph - is a result,
    not an exception. Only `torch.export` failing raises.

    Extra keyword arguments go to the pipeline (`runners_dir`, `artifacts_dir`,
    `threads`, `warmup_iterations`, `measured_iterations`, `repetitions`,
    `profile_iterations`, `verbose`).
    """
    exported_program = export_for_analysis(
        model, args=args, kwargs=kwargs,
        dynamic_shapes=dynamic_shapes, strict=strict,
    )
    return analyze_exported_program(
        exported_program,
        args=args,
        kwargs=kwargs,
        name=name or type(_unwrap(model)).__name__,
        description=f"live {type(_unwrap(model)).__name__} exported in-process",
        argmax_dim=argmax_dim,
        **pipeline_options,
    )
