"""The one DelegateDoctor pipeline, shared by every entry point.

A live `nn.Module` and a `model.pt2` converge on a `ModelSpec` holding an
`ExportedProgram`, and from here neither is distinguishable from the other.

The pipeline is a sequence of stages, each of which can pass, fail, or be
honestly reported as unavailable:

    graph inspection -> ExecuTorch lowering -> XNNPACK analysis
      -> device execution -> runtime profiling -> repair matching
      -> host + device verification -> benchmark -> keep or reject

Stopping early is normal. A model that lowers but cannot run on the attached
target still gets its delegation analysed; a model with no matching repair still
gets its hotspots ranked. Only the *repair decision* needs every stage, and it
is exactly as strict as it has always been: correct on the host, correct on the
device, and measurably faster there.

The control flow below is deliberately linear and readable top to bottom.
"""

from __future__ import annotations

import copy
import os
import shutil

import torch

from . import (
    android_setup,
    benchmarking,
    capabilities,
    console_noise,
    delegation,
    device,
    device_verification,
    export_model,
    html_report,
    profiling,
    reporting,
    result as result_module,
)
from .decision import decide_repair
from .repairs import ALL_RULES
from .result import OptimizationResult
from .verification import verify_repair

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)
DEFAULT_RUNNERS_DIR = os.path.join(PROJECT_DIR, "runners")
DEFAULT_ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")

# What an accepted repair is published as, at the top of the run directory.
OPTIMIZED_PTE_NAME = "optimized_model.pte"


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


def save_input_for_device(input_tensor: torch.Tensor, run_dir: str,
                          index: int = 0) -> str:
    """Write one input tensor as a raw fp32 blob.

    `executor_runner --inputs` reads raw tensor data with no header, so the same
    bytes feed the original and repaired models. One file per positional input.
    """
    path = os.path.join(run_dir, f"input{index}.bin")
    input_tensor.detach().numpy().tofile(path)
    return path


def _find_device(runners_dir: str):
    """Locate the Arm target and both runners, without raising.

    A missing device is a missing *capability*, not a broken model, so it is
    reported as a stage outcome rather than an exception.
    """
    try:
        device_info = device.require_device()
        bench_runner = device.find_runner(runners_dir, device.BENCH_RUNNER_NAME)
        etdump_runner = device.find_runner(runners_dir, device.ETDUMP_RUNNER_NAME)
        return device_info, bench_runner, etdump_runner, ""
    except device.DeviceError as error:
        # Keep the first line: the rest is install advice, printed separately.
        return None, None, None, str(error).strip().splitlines()[0]


def run_optimization(
    model_spec,
    runners_dir: str = DEFAULT_RUNNERS_DIR,
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
    warmup_iterations: int = 20,
    measured_iterations: int = 150,
    repetitions: int = 3,
    threads: int = 4,
    profile_iterations: int = 20,
    run_dir: str = None,
    verbose: bool = False,
    quiet: bool = False,
) -> OptimizationResult:
    """Analyze one exported program as far as the stack and device allow.

    `verbose` adds the full section-by-section report to the console and turns
    off upstream-noise filtering; the full report is written to `report.txt` and
    `report.html` either way. `quiet` suppresses DelegateDoctor's own console
    output, and deliberately does *not* hide warnings or errors.

    The only thing this wrapper adds is a scoped filter for a short allowlist of
    known-benign PyTorch/ExecuTorch messages - see `console_noise`. The filter
    is undone on the way out, including when the analysis raises.
    """
    suppressed_noise = []
    with console_noise.suppress_known_warnings(verbose=verbose):
        return _run_analysis(
            model_spec,
            runners_dir=runners_dir,
            artifacts_dir=artifacts_dir,
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
            repetitions=repetitions,
            threads=threads,
            profile_iterations=profile_iterations,
            run_dir=run_dir,
            verbose=verbose,
            quiet=quiet,
            suppressed_noise=suppressed_noise,
        )


def _run_analysis(
    model_spec,
    runners_dir: str,
    artifacts_dir: str,
    warmup_iterations: int,
    measured_iterations: int,
    repetitions: int,
    threads: int,
    profile_iterations: int,
    run_dir: str,
    verbose: bool,
    quiet: bool,
    suppressed_noise: list,
) -> OptimizationResult:
    """The pipeline itself. See `run_optimization` for the public contract."""
    report_parts = []

    def emit(section: str) -> None:
        """Keep a section for report.txt, and print it only when asked.

        The console default is the short summary at the end: the long form is
        preserved in the artifacts, where it is easier to read anyway.
        """
        section = (section or "").rstrip("\n")
        if not section.strip():
            return
        if verbose and not quiet:
            print(section)
        report_parts.append(section)

    def note(line: str) -> None:
        """Progress chatter that does not belong in the saved report.

        Device stages take minutes, so these stay on by default - silence for
        two minutes is worse than four short lines.
        """
        if not quiet:
            print(line)

    outcome = OptimizationResult(
        status=result_module.ANALYSIS_COMPLETE,
        model_name=model_spec.name,
        description=model_spec.description,
        # Copied so the report can describe a repair without importing the
        # rules. Presentation metadata only; nothing here affects a decision.
        repair_catalog={
            rule.RULE_ID: {
                "title": rule.RULE_TITLE,
                "rewrite": rule.describe_rewrite(),
                "matches": rule.matches_portable_kernel,
                "flow_before": getattr(rule, "FLOW_BEFORE", None),
                "flow_after": getattr(rule, "FLOW_AFTER", None),
            }
            for rule in ALL_RULES
        },
    )
    # Getting here at all means an ExportedProgram exists: either torch.export
    # captured the live model, or torch.export.load read a .pt2.
    outcome.record(result_module.EXPORT, result_module.PASS)

    if run_dir is None:
        run_dir = next_run_directory(artifacts_dir)
    outcome.run_dir = run_dir
    before_dir = os.path.join(run_dir, "before")
    after_dir = os.path.join(run_dir, "after")

    args = tuple(model_spec.example_args)
    kwargs = dict(model_spec.example_kwargs)

    def finish(status: str) -> OptimizationResult:
        """Fill in the stages that never ran, write artifacts, and return."""
        for name in result_module.STAGE_ORDER:
            if outcome.stage(name) is None:
                outcome.record(name, result_module.NOT_RUN)
        outcome.status = status
        emit(reporting.format_pipeline(outcome.stages))
        emit(reporting.format_result(outcome))
        outcome.report_text = "\n".join(report_parts)
        reporting.write_text(outcome.report_text, os.path.join(run_dir, "report.txt"))

        # The HTML report is a presentation layer over what is already here:
        # it re-measures nothing and re-runs nothing.
        try:
            outcome.report_path = html_report.generate_html_report(
                outcome, run_dir, android_setup.SUPPORTED_EXECUTORCH_VERSION
            )
        except Exception as error:                       # pragma: no cover
            # A formatting bug must never discard a completed analysis.
            note(f"(HTML report could not be written: "
                 f"{type(error).__name__}: {error})")

        reporting.write_json(outcome.to_dict(), os.path.join(run_dir, "result.json"))
        if suppressed_noise:
            reporting.write_text(
                console_noise.describe_policy()
                + "\n\nSuppressed native stderr\n"
                + "\n".join(suppressed_noise),
                os.path.join(run_dir, "runtime.log"),
            )
        if not quiet:
            print(reporting.format_summary(outcome))
        return outcome

    # --- target and tooling, if any ----------------------------------------
    device_info, bench_runner, etdump_runner, device_reason = _find_device(runners_dir)
    if device_info is not None:
        outcome.device_description = device_info.short_description()
        outcome.device_is_emulator = device_info.is_emulator

    emit(reporting.format_header(
        model_name=model_spec.name,
        input_shape=capabilities.describe_arguments(args, kwargs),
        target_description=(device_info.short_description() if device_info
                            else "no Arm64 target attached"),
    ))

    # --- the pristine baseline ---------------------------------------------
    # `model_spec.exported_program` is the reference the repair is judged
    # against, so it is never lowered and never repaired. Lowering and the
    # repair rules each get their own deep copy: `to_edge_transform_and_lower`
    # is free to mutate what it is given, and the rules certainly do.
    baseline_program = model_spec.exported_program
    exported_for_baseline = copy.deepcopy(baseline_program)
    exported_for_repair = copy.deepcopy(baseline_program)
    node_count = len(list(baseline_program.graph.nodes))
    outcome.record(result_module.GRAPH, result_module.PASS,
                   f"{node_count} graph nodes")

    # --- lower the original graph ------------------------------------------
    note("\nLowering with XNNPACK...")
    try:
        before_export = export_model.lower_and_write(
            exported_for_baseline, os.path.join(before_dir, "model.pte")
        )
    except Exception as error:
        # The graph exported fine; ExecuTorch is what could not take it. That
        # is a finding about this deployment path, not a failed model.
        outcome.record(
            result_module.LOWERING, result_module.FAILED,
            f"{type(error).__name__}: {str(error)[:400]}",
        )
        emit(reporting.format_lowering_failure(
            model_spec.name, error, android_setup.SUPPORTED_EXECUTORCH_VERSION
        ))
        return finish(result_module.EXECUTORCH_LOWERING_UNSUPPORTED)

    outcome.record(result_module.LOWERING, result_module.PASS)
    export_model.save_readable_graphs(before_export, before_dir)

    before_delegation = delegation.analyze_delegation(before_export.edge_program_manager)
    outcome.before_delegation = before_delegation
    outcome.record(
        result_module.DELEGATION, result_module.PASS,
        f"{before_delegation.portable_op_total} portable of "
        f"{before_delegation.total_ops} ops",
    )

    # --- repair matching is static, so it runs with or without a device -----
    detections = {rule.RULE_ID: rule.detect(exported_for_repair) for rule in ALL_RULES}
    outcome.detections = detections
    applicable = [(rule, detections[rule.RULE_ID]) for rule in ALL_RULES
                  if detections[rule.RULE_ID].applies]
    outcome.record(
        result_module.REPAIR,
        result_module.PASS if applicable else result_module.NONE_FOUND,
        ", ".join(rule.RULE_ID for rule, _ in applicable),
    )

    # --- can this model reach the device at all? ---------------------------
    def stop_before_device(status: str, reason: str, blocked: bool):
        """Report everything static, and be explicit that no repair was tried.

        A matched pattern is not a repair: accepting one needs the device
        benchmark, so without a device the rule stops at "candidate observed".
        """
        outcome.record(result_module.DEVICE, status, reason)
        if applicable:
            outcome.record(
                result_module.REPAIR, result_module.PASS,
                ", ".join(rule.RULE_ID for rule, _ in applicable)
                + " matched; not applied without a device benchmark",
            )
        emit(reporting.format_static_analysis(
            before_delegation, detections, status, reason,
            matched=[rule.RULE_ID for rule, _ in applicable],
        ))
        return finish(_static_status(before_delegation, device_blocked=blocked))

    if device_info is None:
        return stop_before_device(result_module.UNAVAILABLE, device_reason, False)

    input_capability = capabilities.assess_inputs(args, kwargs)
    if not input_capability:
        return stop_before_device(
            result_module.UNSUPPORTED, input_capability.reason, True
        )

    outcome.record(result_module.DEVICE, result_module.PASS)

    # --- the same frozen inputs, staged for the device ---------------------
    device_input_paths = [
        save_input_for_device(tensor, run_dir, index)
        for index, tensor in enumerate(args)
    ]

    # --- profile the original on the device --------------------------------
    note("Profiling original on device...")
    before_profile = profiling.profile_model(
        pte_path=before_export.pte_path,
        input_path=device_input_paths[0],
        etdump_runner_path=etdump_runner,
        output_etdump_path=os.path.join(before_dir, "trace.etdump"),
        label="before",
        serial=device_info.serial,
        iterations=profile_iterations,
        threads=threads,
    )
    outcome.before_profile = before_profile
    outcome.record(result_module.PROFILING, result_module.PASS)
    reporting.write_json(before_profile.to_dict(), os.path.join(before_dir, "profile.json"))

    emit(reporting.format_analysis(before_delegation, before_profile))

    # Link measured hotspots to whichever rule can repair them.
    repairable_kernels = {}
    for rule, _ in applicable:
        for kernel in before_profile.portable_kernels:
            if rule.matches_portable_kernel(kernel.name):
                repairable_kernels[kernel.name] = rule.RULE_ID

    emit(reporting.format_hotspots(before_profile, repairable_kernels))

    if not applicable:
        # Analysis succeeded; there is simply no rule for what it found. That is
        # a useful result, not a tool failure.
        emit(reporting.format_no_repair(before_profile))
        emit(reporting.format_declined_repairs(detections))
        return finish(
            result_module.NO_REPAIR_REQUIRED if not before_profile.portable_kernels
            else result_module.NO_REPAIR_AVAILABLE
        )

    # --- apply every applicable rule and re-export -------------------------
    repaired_counts = {}
    for rule, found in applicable:
        emit(reporting.format_detection(rule, found))
        repaired_counts[rule.RULE_ID] = rule.apply(exported_for_repair)
        emit(reporting.format_repair(rule, repaired_counts[rule.RULE_ID]))
    outcome.repairs_applied = repaired_counts
    report_parts.append(f"\nConfiguration: {model_spec.description}")

    after_export = export_model.lower_and_write(
        exported_for_repair, os.path.join(after_dir, "model.pte")
    )
    export_model.save_readable_graphs(after_export, after_dir)
    after_delegation = delegation.analyze_delegation(after_export.edge_program_manager)
    outcome.after_delegation = after_delegation

    note("Profiling repaired on device...")
    after_profile = profiling.profile_model(
        pte_path=after_export.pte_path,
        input_path=device_input_paths[0],
        etdump_runner_path=etdump_runner,
        output_etdump_path=os.path.join(after_dir, "trace.etdump"),
        label="after",
        serial=device_info.serial,
        iterations=profile_iterations,
        threads=threads,
    )
    outcome.after_profile = after_profile
    reporting.write_json(after_profile.to_dict(), os.path.join(after_dir, "profile.json"))

    emit(reporting.format_delegation_change(
        before_delegation, after_delegation, before_profile, after_profile
    ))

    # --- numerical gate ----------------------------------------------------
    # The reference output comes from the pristine baseline graph, which has
    # been neither lowered nor repaired - so this is what the model meant
    # before DelegateDoctor touched anything.
    note("Verifying on host...")
    eager_output = model_spec.call_baseline()
    output_capability = capabilities.assess_output(eager_output)

    if not output_capability:
        # The repair may well be fine, but nothing here can prove it. Saying so
        # is the only honest option: an unverified repair is never accepted.
        outcome.record(result_module.VERIFICATION, result_module.UNSUPPORTED,
                       output_capability.reason)
        emit(reporting.format_unverifiable(output_capability.reason))
        return finish(result_module.DEVICE_EXECUTION_UNSUPPORTED)

    eager_tensor = capabilities.first_output_tensor(eager_output)
    # ExecuTorch's C++ runtime narrates a CPU probe on first use, straight to
    # fd 2. Scoped tightly around the two host executions that trigger it;
    # anything unrecognised on stderr is replayed untouched.
    with console_noise.filter_native_stderr(verbose=verbose,
                                            suppressed=suppressed_noise):
        original_output = export_model.run_on_host(before_export.pte_path, args)[0]
        repaired_output = export_model.run_on_host(after_export.pte_path, args)[0]

    verification_result = verify_repair(
        original_output=original_output,
        repaired_output=repaired_output,
        eager_output=eager_tensor,
        argmax_dim=model_spec.argmax_dim,
    )
    outcome.host_verification = verification_result

    # Device-side verification: run both .pte files on the Android target and
    # compare the tensors it actually produced. This is a separate, untimed
    # invocation - the benchmark below still writes no output tensors, so this
    # cannot affect latency numbers.
    note("Verifying on device...")
    try:
        device_result = device_verification.run_device_verification(
            before_pte_path=before_export.pte_path,
            after_pte_path=after_export.pte_path,
            input_paths=device_input_paths,
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
    outcome.device_verification = device_result
    outcome.record(
        result_module.VERIFICATION,
        result_module.PASS if verification_result.passed and device_result.passed
        else result_module.FAILED,
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
    note("Benchmarking on device...")
    benchmark_result = benchmarking.benchmark_before_after(
        before_pte_path=before_export.pte_path,
        after_pte_path=after_export.pte_path,
        input_paths=device_input_paths,
        bench_runner_path=bench_runner,
        device_info=device_info,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        repetitions=repetitions,
        threads=threads,
    )
    outcome.benchmark = benchmark_result
    outcome.record(result_module.BENCHMARK, result_module.PASS)
    reporting.write_json(
        benchmark_result.to_dict(), os.path.join(run_dir, "benchmark.json")
    )
    emit(reporting.format_benchmark(benchmark_result))

    # --- accept or reject --------------------------------------------------
    # Unchanged policy: correct on the host, correct on the device, and faster
    # on the device. Nothing about being "more delegated" enters this.
    decision = decide_repair(
        host_verification_passed=verification_result.passed,
        device_verification_passed=device_result.passed,
        before_latency_ms=benchmark_result.before.p50_ms,
        after_latency_ms=benchmark_result.after.p50_ms,
    )
    outcome.decision = decision
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
    # Also at a stable path, so scripts do not have to find the newest run_NNN.
    reporting.write_json(results, os.path.join(PROJECT_DIR, "results", "latest.json"))

    if decision.accepted:
        # Only a repair the device proved worthwhile is published under the
        # name a user is meant to ship. A rejected repair stays in after/.
        optimized_path = os.path.join(run_dir, OPTIMIZED_PTE_NAME)
        shutil.copyfile(after_export.pte_path, optimized_path)
        outcome.output_pte = optimized_path

    return finish(result_module.REPAIR_ACCEPTED if decision.accepted
                  else result_module.REPAIR_REJECTED)


def _static_status(before_delegation, device_blocked: bool = False) -> str:
    """The outcome when the run stopped before the device stages.

    A fully delegated graph is a finished answer even with no device attached:
    there is no portable operator left to repair.
    """
    if before_delegation.portable_op_total == 0:
        return result_module.FULLY_DELEGATED
    if device_blocked:
        return result_module.DEVICE_EXECUTION_UNSUPPORTED
    return result_module.ANALYSIS_COMPLETE
