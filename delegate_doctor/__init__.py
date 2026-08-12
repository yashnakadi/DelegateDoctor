"""DelegateDoctor - analyze and repair the ExecuTorch/XNNPACK deployment path.

    from delegate_doctor import optimize

    result = optimize(model, args=(example_input,))
    print(result.status)

If PyTorch can export your model, DelegateDoctor can inspect the exported graph.
It reports how that graph lowers to ExecuTorch, which operators XNNPACK refuses,
and where the runtime actually goes on an Arm64 target. If it then recognises a
proven repair, it verifies and benchmarks that repair on the device and keeps it
only when it is both correct and measurably faster.

Analysis is the product; optimization is an additional capability. A model with
no matching repair, or one the attached target cannot run, is still analyzed as
far as the stack allows - and the result says exactly where it stopped.

Start at `delegate_doctor.pipeline.run_optimization` to read the stages top to
bottom.
"""

from . import console_noise

# Two torch log records are emitted *while* ExecuTorch imports, so this has to
# happen before the import below - there is no later hook. It installs a
# message-matching filter on two named loggers and touches nothing else; in
# particular it does not alter the `warnings` module, so importing
# DelegateDoctor cannot change how warnings behave in your program. Undo it with
# `console_noise.restore_import_logging()`, or set
# DELEGATE_DOCTOR_VERBOSE_IMPORTS=1 to skip it.
console_noise.quieten_import_logging()

from .api import ExportFailed, analyze_exported_program, export_for_analysis, optimize
from .result import OptimizationResult

__version__ = "0.1.0"

# Deliberately small. Note `export_for_analysis` rather than `export_model`:
# `delegate_doctor.export_model` is a module, and a function of that name here
# would shadow it for anyone doing `from delegate_doctor import export_model`.
__all__ = [
    "optimize",
    "analyze_exported_program",
    "export_for_analysis",
    "ExportFailed",
    "OptimizationResult",
]
