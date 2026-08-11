"""The `delegate-doctor` command line interface.

One command, `doctor`, which runs the whole workflow:

    build model -> export -> lower with XNNPACK -> analyze delegation
      -> profile on device -> rank hotspots -> detect DD-001 -> apply DD-001
      -> re-export -> verify numerically -> benchmark -> accept or reject

The control flow below is deliberately linear and readable top to bottom.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import torch

from . import (
    android_setup,
    benchmarking,
    delegation,
    device,
    device_verification,
    export_model,
    models,
    profiling,
    reporting,
)
from .decision import decide_repair
from .repairs import ALL_RULES
from .verification import verify_repair

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)
DEFAULT_RUNNERS_DIR = os.path.join(PROJECT_DIR, "runners")
DEFAULT_ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")

# Demo workloads shipped with DelegateDoctor: one example file per
# architecture. All six are real segmentation models that produce the DD-001
# pattern naturally; see delegate_doctor/models.py.
BUILTIN_EXAMPLES = {
    name: os.path.join(PROJECT_DIR, "examples", f"{name}.py")
    for name in models.MODEL_NAMES
}


def load_model_spec(model_argument: str) -> export_model.ModelSpec:
    """Load a ModelSpec from a built-in example name or a Python file path.

    The file must define `build_model()` returning a ModelSpec. Keeping models
    outside the package means DelegateDoctor never depends on any particular
    model library.
    """
    if model_argument in BUILTIN_EXAMPLES:
        path = BUILTIN_EXAMPLES[model_argument]
    else:
        path = model_argument

    if not os.path.isfile(path):
        available = "\n".join(f"  {name}" for name in BUILTIN_EXAMPLES)
        raise SystemExit(
            f"Unknown model: {model_argument}\n"
            f"\n"
            f"Available models:\n{available}\n"
            f"\n"
            f"Or pass the path to a Python file that defines build_model()."
        )

    module_name = "delegate_doctor_model"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    # Register before executing: torch.export traces through the model's own
    # module and re-imports it by name, which fails if it is not in sys.modules.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "build_model"):
        raise SystemExit(f"{path} does not define a build_model() function.")

    model_spec = module.build_model()
    if not isinstance(model_spec, export_model.ModelSpec):
        raise SystemExit(
            f"{path}: build_model() must return a delegate_doctor.export_model.ModelSpec."
        )
    return model_spec


def next_run_directory(artifacts_dir: str) -> str:
    """Allocate artifacts/run_001, run_002, ... so runs never overwrite."""
    os.makedirs(artifacts_dir, exist_ok=True)
    existing = [name for name in os.listdir(artifacts_dir) if name.startswith("run_")]
    run_number = len(existing) + 1
    while True:
        candidate = os.path.join(artifacts_dir, f"run_{run_number:03d}")
        if not os.path.exists(candidate):
            os.makedirs(candidate)
            return candidate
        run_number += 1


def save_input_for_device(input_tensor: torch.Tensor, run_dir: str) -> str:
    """Write the benchmark input as a raw fp32 blob.

    `executor_runner --inputs` reads raw tensor data with no header, so the
    same bytes feed both the before and after models.
    """
    path = os.path.join(run_dir, "input.bin")
    input_tensor.numpy().tofile(path)
    return path


def run_doctor(
    model_argument: str,
    runners_dir: str,
    artifacts_dir: str,
    warmup_iterations: int,
    measured_iterations: int,
    repetitions: int,
    threads: int,
    profile_iterations: int,
    seed: int,
) -> int:
    """Run the full workflow. Returns a process exit code."""
    report_parts = []

    def emit(section: str) -> None:
        """Print a section immediately and keep it for report.txt.

        Sections already begin with their own blank line, so trailing newlines
        are stripped to avoid double-spacing the console.
        """
        section = section.rstrip("\n")
        print(section)
        report_parts.append(section)

    # --- target and tooling ------------------------------------------------
    device_info = device.require_device()
    bench_runner = device.find_runner(runners_dir, device.BENCH_RUNNER_NAME)
    etdump_runner = device.find_runner(runners_dir, device.ETDUMP_RUNNER_NAME)

    model_spec = load_model_spec(model_argument)
    run_dir = next_run_directory(artifacts_dir)
    before_dir = os.path.join(run_dir, "before")
    after_dir = os.path.join(run_dir, "after")

    input_shape = "x".join(str(int(s)) for s in model_spec.example_inputs[0].shape)
    emit(reporting.format_header(
        model_name=model_spec.name,
        input_shape=input_shape,
        target_description=device_info.short_description(),
    ))

    # --- export and lower the original model -------------------------------
    print("\nExporting and lowering with XNNPACK...")
    original_exported = export_model.export_to_aten(
        model_spec.model, model_spec.example_inputs
    )
    # DD-001 mutates the graph, so keep an untouched copy for the "before" path.
    exported_for_repair = copy.deepcopy(original_exported)

    before_export = export_model.lower_and_write(
        original_exported, os.path.join(before_dir, "model.pte")
    )
    export_model.save_readable_graphs(before_export, before_dir)
    before_delegation = delegation.analyze_delegation(before_export.edge_program_manager)

    # --- one deterministic input, shared by every measurement --------------
    torch.manual_seed(seed)
    benchmark_input = torch.randn(*model_spec.example_inputs[0].shape)
    device_input_path = save_input_for_device(benchmark_input, run_dir)

    # --- profile the original on the device --------------------------------
    print("Profiling original on device...")
    before_profile = profiling.profile_model(
        pte_path=before_export.pte_path,
        input_path=device_input_path,
        etdump_runner_path=etdump_runner,
        output_etdump_path=os.path.join(before_dir, "trace.etdump"),
        label="before",
        serial=device_info.serial,
        iterations=profile_iterations,
        threads=threads,
    )
    reporting.write_json(before_profile.to_dict(), os.path.join(before_dir, "profile.json"))

    emit(reporting.format_analysis(before_delegation, before_profile))

    # --- detect: try each repair rule in turn ------------------------------
    detections = [(rule, rule.detect(exported_for_repair)) for rule in ALL_RULES]
    applicable = [(rule, found) for rule, found in detections if found.applies]

    # Link measured hotspots to whichever rule can repair them.
    repairable_kernels = {}
    for rule, _ in applicable:
        for kernel in before_profile.portable_kernels:
            if rule.matches_portable_kernel(kernel.name):
                repairable_kernels[kernel.name] = rule.RULE_ID

    emit(reporting.format_hotspots(before_profile, repairable_kernels))

    if not applicable:
        for rule, found in detections:
            emit(reporting.format_detection(rule, found))
        emit("\nNo known repair pattern found in this model. Nothing to repair.\n")
        reporting.write_text("\n".join(report_parts), os.path.join(run_dir, "report.txt"))
        print(f"Artifacts: {run_dir}")
        return 0

    # --- apply every applicable rule and re-export -------------------------
    repaired_counts = {}
    for rule, found in applicable:
        emit(reporting.format_detection(rule, found))
        repaired_counts[rule.RULE_ID] = rule.apply(exported_for_repair)
        emit(reporting.format_repair(rule, repaired_counts[rule.RULE_ID]))
    repaired_count = sum(repaired_counts.values())
    report_parts.append(f"\nConfiguration: {model_spec.description}")

    after_export = export_model.lower_and_write(
        exported_for_repair, os.path.join(after_dir, "model.pte")
    )
    export_model.save_readable_graphs(after_export, after_dir)
    after_delegation = delegation.analyze_delegation(after_export.edge_program_manager)

    print("Profiling repaired on device...")
    after_profile = profiling.profile_model(
        pte_path=after_export.pte_path,
        input_path=device_input_path,
        etdump_runner_path=etdump_runner,
        output_etdump_path=os.path.join(after_dir, "trace.etdump"),
        label="after",
        serial=device_info.serial,
        iterations=profile_iterations,
        threads=threads,
    )
    reporting.write_json(after_profile.to_dict(), os.path.join(after_dir, "profile.json"))

    emit(reporting.format_delegation_change(
        before_delegation, after_delegation, before_profile, after_profile
    ))

    # --- numerical gate ----------------------------------------------------
    print("Verifying on host...")
    with torch.no_grad():
        eager_output = model_spec.model(benchmark_input)
    if isinstance(eager_output, (tuple, list)):
        eager_output = eager_output[0]

    original_output = export_model.run_on_host(before_export.pte_path, (benchmark_input,))[0]
    repaired_output = export_model.run_on_host(after_export.pte_path, (benchmark_input,))[0]

    verification_result = verify_repair(
        original_output=original_output,
        repaired_output=repaired_output,
        eager_output=eager_output,
        argmax_dim=model_spec.argmax_dim,
    )

    # Device-side verification: run both .pte files on the Android target and
    # compare the tensors it actually produced. This is a separate, untimed
    # invocation - the benchmark below still writes no output tensors, so this
    # cannot affect latency numbers.
    print("Verifying on device...")
    try:
        device_result = device_verification.run_device_verification(
            before_pte_path=before_export.pte_path,
            after_pte_path=after_export.pte_path,
            input_path=device_input_path,
            bench_runner_path=bench_runner,
            original_host_output=original_output,
            repaired_host_output=repaired_output,
            output_dir=run_dir,
            serial=device_info.serial,
            argmax_dim=model_spec.argmax_dim,
            threads=threads,
        )
    except device_verification.DeviceVerificationError as error:
        device_result = device_verification.DeviceVerificationResult(
            passed=False,
            failure_reasons=[str(error)],
            error=str(error),
        )

    reporting.write_json(
        {
            "host": verification_result.to_dict(),
            "device": device_result.to_dict(),
        },
        os.path.join(run_dir, "verification.json"),
    )
    emit(reporting.format_verification(verification_result, device_result))

    # --- performance gate --------------------------------------------------
    print("Benchmarking on device...")
    benchmark_result = benchmarking.benchmark_before_after(
        before_pte_path=before_export.pte_path,
        after_pte_path=after_export.pte_path,
        input_path=device_input_path,
        bench_runner_path=bench_runner,
        device_info=device_info,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        repetitions=repetitions,
        threads=threads,
    )
    reporting.write_json(
        benchmark_result.to_dict(), os.path.join(run_dir, "benchmark.json")
    )
    emit(reporting.format_benchmark(benchmark_result))

    # --- accept or reject --------------------------------------------------
    decision = decide_repair(
        host_verification_passed=verification_result.passed,
        device_verification_passed=device_result.passed,
        before_latency_ms=benchmark_result.before.p50_ms,
        after_latency_ms=benchmark_result.after.p50_ms,
    )
    emit(reporting.format_decision(decision, device_info.short_description()))

    results = reporting.build_results_json(
        rules_applied=repaired_counts,
        model_name=model_spec.name,
        device_description=device_info.describe(),
        before_delegation=before_delegation,
        after_delegation=after_delegation,
        before_profile=before_profile,
        after_profile=after_profile,
        verification_result=verification_result,
        device_verification_result=device_result,
        benchmark_result=benchmark_result,
        decision=decision,
    )
    reporting.write_json(results, os.path.join(run_dir, "results.json"))
    reporting.write_text("\n".join(report_parts), os.path.join(run_dir, "report.txt"))
    # Also at a stable path, so scripts do not have to find the newest run_NNN.
    reporting.write_json(results, os.path.join(PROJECT_DIR, "results", "latest.json"))

    print(f"Artifacts: {run_dir}")
    if decision.accepted:
        print(f"Repaired model: {after_export.pte_path}")
    return 0 if decision.accepted else 1


def _quieten_known_upstream_warnings() -> None:
    """Filter one specific, harmless upstream warning.

    torch.export emits this FutureWarning from inside pytree once per traced
    submodule, so a single export prints it dozens of times and buries the
    report. It is matched by exact message text: nothing else is silenced, and
    any warning we have not seen before still reaches the user.
    """
    # Matched loosely enough to survive the backticks in the upstream text, but
    # still tied to this one message: it must mention both treespec and LeafSpec.
    warnings.filterwarnings(
        "ignore",
        message=r".*treespec.*LeafSpec.*",
        category=FutureWarning,
    )


def main(argv: Optional[list] = None) -> int:
    _quieten_known_upstream_warnings()
    parser = argparse.ArgumentParser(
        prog="delegate-doctor",
        description=(
            "Find and repair expensive ExecuTorch/XNNPACK fallback operations, "
            "then keep the repair only if it is correct and faster on Arm64."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor", help="run the full analyze-repair-verify-benchmark workflow"
    )
    doctor.add_argument(
        "model",
        metavar="MODEL",
        help=(
            "one of: " + ", ".join(models.MODEL_NAMES)
            + "; or the path to a Python file defining build_model()"
        ),
    )
    doctor.add_argument("--runners-dir", default=DEFAULT_RUNNERS_DIR,
                        help="directory holding the arm64 executor_runner binaries")
    doctor.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    doctor.add_argument("--warmup", type=int, default=20,
                        help="warmup iterations per repetition")
    doctor.add_argument("--iters", type=int, default=150,
                        help="measured iterations per repetition")
    doctor.add_argument("--reps", type=int, default=3,
                        help="interleaved before/after repetitions")
    doctor.add_argument("--threads", type=int, default=4,
                        help="CPU threads used on the device")
    doctor.add_argument("--profile-iters", type=int, default=20,
                        help="iterations traced when profiling")
    doctor.add_argument("--seed", type=int, default=1234,
                        help="seed for the deterministic benchmark input")

    setup = subparsers.add_parser(
        "setup-android",
        help="download the pinned ExecuTorch source and build the Arm64 runners",
    )
    setup.add_argument("--runners-dir", default=DEFAULT_RUNNERS_DIR,
                       help="where the built runner binaries are installed")
    setup.add_argument("--rebuild", action="store_true",
                       help="rebuild even if the runners already exist")
    setup.add_argument("--jobs", type=int, default=10,
                       help="parallel compile jobs")

    args = parser.parse_args(argv)

    if args.command == "setup-android":
        try:
            return android_setup.setup_android_runners(
                project_root=Path(PROJECT_DIR),
                runners_dir=Path(args.runners_dir),
                rebuild=args.rebuild,
                parallel_jobs=args.jobs,
            )
        except android_setup.SetupError as error:
            print(f"\n{error}", file=sys.stderr)
            return 2

    if args.command == "doctor":
        try:
            return run_doctor(
                model_argument=args.model,
                runners_dir=args.runners_dir,
                artifacts_dir=args.artifacts_dir,
                warmup_iterations=args.warmup,
                measured_iterations=args.iters,
                repetitions=args.reps,
                threads=args.threads,
                profile_iterations=args.profile_iters,
                seed=args.seed,
            )
        except device.DeviceError as error:
            print(f"\nDevice error:\n{error}", file=sys.stderr)
            return 2

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
