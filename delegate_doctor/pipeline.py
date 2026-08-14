"""The one DelegateDoctor pipeline, shared by every entry point.

Every entry point - the CLI's `model.py`, the Python API's live `nn.Module` -
converges on a `ModelSpec` holding an `ExportedProgram`, and from here they are
indistinguishable.

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
    model_exploration,
    profiling,
    repair_loop,
    repair_opportunity,
    reporting,
    result as result_module,
    target_selection,
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


def _find_device(runners_dir: str, target_preference: str = "auto",
                 target_serial: str = None, interactive: bool = False):
    """Choose the Arm target and locate both runners, without raising.

    A missing device is a missing *capability*, not a broken model, so it is
    reported as a stage outcome rather than an exception.

    The chosen target's serial is returned inside `DeviceInfo` and is then
    threaded through profiling, device verification and benchmarking, so a
    before/after comparison can never straddle two machines.
    """
    try:
        target = target_selection.select_target(
            preference=target_preference,
            serial=target_serial,
            interactive=interactive,
            announce=lambda *args, **kwargs: None,
        )
        bench_runner = device.find_runner(runners_dir, device.BENCH_RUNNER_NAME)
        etdump_runner = device.find_runner(runners_dir, device.ETDUMP_RUNNER_NAME)
        return target.info, bench_runner, etdump_runner, ""
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
    target_preference: str = "auto",
    target_serial: str = None,
    interactive: bool = False,
    # Experimental AI repair is opt-in. Without it the pipeline stops after
    # the known repairs, which is the whole default product: diagnose, apply
    # proven DDs, verify, benchmark.
    ai_repair: bool = False,
    prompt=input,
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
            target_preference=target_preference,
            target_serial=target_serial,
            interactive=interactive,
            ai_repair=ai_repair,
            prompt=prompt,
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
    target_preference: str,
    target_serial: str,
    interactive: bool,
    ai_repair: bool,
    prompt,
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
        if outcome.opportunity is not None:
            emit(repair_opportunity.format_report_section(outcome.opportunity))
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
    device_info, bench_runner, etdump_runner, device_reason = _find_device(
        runners_dir, target_preference=target_preference,
        target_serial=target_serial, interactive=interactive)
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

    # Everything a human needs to judge whether a repair is worth pursuing,
    # derived once. The consent screen, report.txt and report.html all render
    # from this object, so they cannot disagree about a percentage.
    outcome.opportunity = repair_opportunity.build_summary(
        profile=before_profile,
        delegation=before_delegation,
        target=outcome.device_description,
        catalog_match=(", ".join(rule.RULE_ID for rule, _ in applicable)
                       if applicable else "None"),
    )

    # --- the host reference, established once --------------------------------
    # Every candidate - the first and the fifth - is verified against what the
    # ORIGINAL model produced. Comparing each repair only against the one
    # before it would let small errors accumulate into a wrong model while
    # every individual step looked fine.
    note("Verifying on host...")
    eager_output = model_spec.call_baseline()
    output_capability = capabilities.assess_output(eager_output)
    if not output_capability:
        # A repair may well be fine, but nothing here can prove it, and an
        # unverified repair is never accepted.
        outcome.record(result_module.VERIFICATION, result_module.UNSUPPORTED,
                       output_capability.reason)
        emit(reporting.format_unverifiable(output_capability.reason))
        return finish(result_module.DEVICE_EXECUTION_UNSUPPORTED)

    eager_tensor = capabilities.first_output_tensor(eager_output)
    with console_noise.filter_native_stderr(verbose=verbose,
                                            suppressed=suppressed_noise):
        original_host_output = export_model.run_on_host(
            before_export.pte_path, args)[0]

    # --- the iterative repair loop -------------------------------------------
    machinery = _RepairMachinery(
        outcome=outcome, run_dir=run_dir, model_spec=model_spec, args=args,
        device_info=device_info, bench_runner=bench_runner,
        etdump_runner=etdump_runner, device_input_paths=device_input_paths,
        original_pte_path=before_export.pte_path,
        original_host_output=original_host_output, eager_tensor=eager_tensor,
        profile_iterations=profile_iterations, threads=threads,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations, repetitions=repetitions,
        verbose=verbose, suppressed_noise=suppressed_noise,
        interactive=interactive, ai_repair=ai_repair, prompt=prompt,
        emit=emit, note=note,
    )
    history = machinery.run(
        current_program=exported_for_repair,
        current_export=before_export,
        current_profile=before_profile,
        current_delegation=before_delegation,
    )
    outcome.repair_history = history
    # The run-level opportunity summary reports what became of AI across the
    # whole sequence, so a run that asked three times and was declined once
    # does not read as "not requested".
    outcome.opportunity.ai_status = _ai_status_from(history)
    outcome.repairs_applied = {attempt.label: 1 for attempt in history.accepted}

    emit(repair_loop.format_history(history))
    reporting.write_json(history.to_dict(),
                         os.path.join(run_dir, "repair_history.json"))

    if not history.any_accepted:
        # Nothing survived its gates. Whether that is "nothing to repair" or
        # "nothing worked" is a real distinction, and both are honest answers.
        outcome.after_profile = machinery.current_profile
        outcome.after_delegation = machinery.current_delegation
        if history.rejected_count:
            return finish(result_module.REPAIR_REJECTED)
        return finish(
            result_module.NO_REPAIR_REQUIRED
            if not before_profile.portable_kernels
            else result_module.NO_REPAIR_AVAILABLE
        )

    outcome.after_profile = machinery.current_profile
    outcome.after_delegation = machinery.current_delegation
    outcome.host_verification = machinery.last_host_verification
    outcome.device_verification = machinery.last_device_verification
    outcome.decision = machinery.last_decision

    emit(reporting.format_delegation_change(
        before_delegation, machinery.current_delegation,
        before_profile, machinery.current_profile
    ))

    # --- the cumulative headline ---------------------------------------------
    # Per-step benchmarks are incremental by design, and they are measured in
    # separate interleaved invocations. Chaining them arithmetically would
    # accumulate whatever thermal drift each one cancelled internally, so the
    # original-to-final number is measured directly instead - once, at the end.
    final_benchmark = machinery.benchmark_original_against_final()
    if final_benchmark is not None:
        outcome.benchmark = final_benchmark
        history.original_latency_ms = final_benchmark.before.p50_ms
        history.final_latency_ms = final_benchmark.after.p50_ms
        reporting.write_json(final_benchmark.to_dict(),
                             os.path.join(run_dir, "benchmark.json"))
        emit(reporting.format_benchmark(final_benchmark))

    history.original_operator_delegation = \
        before_delegation.operator_delegation_fraction
    history.original_runtime_delegation = \
        before_profile.runtime_delegation_fraction
    history.final_operator_delegation = \
        machinery.current_delegation.operator_delegation_fraction
    history.final_runtime_delegation = \
        machinery.current_profile.runtime_delegation_fraction
    reporting.write_json(history.to_dict(),
                         os.path.join(run_dir, "repair_history.json"))

    results = reporting.build_results_json(
        rules_applied={attempt.label: 1 for attempt in history.accepted},
        model_name=model_spec.name,
        device_description=device_info.describe(),
        before_delegation=before_delegation,
        after_delegation=machinery.current_delegation,
        before_profile=before_profile,
        after_profile=machinery.current_profile,
        verification_result=machinery.last_host_verification,
        device_verification_result=machinery.last_device_verification,
        benchmark_result=outcome.benchmark,
        decision=machinery.last_decision,
    )
    results["repair_history"] = history.to_dict()
    reporting.write_json(results, os.path.join(run_dir, "results.json"))
    reporting.write_json(results, os.path.join(PROJECT_DIR, "results", "latest.json"))

    # Only a program the device proved worthwhile is published under the name a
    # user is meant to ship.
    optimized_path = os.path.join(run_dir, OPTIMIZED_PTE_NAME)
    shutil.copyfile(machinery.current_export.pte_path, optimized_path)
    outcome.output_pte = optimized_path

    return finish(result_module.REPAIRS_ACCEPTED if history.accepted_count > 1
                  else result_module.REPAIR_ACCEPTED)


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


# What the run actually concluded. The previous text asserted that portable
# runtime was below the AI threshold and printed that verbatim on a run whose
# portable runtime was 60.8% - because it was a fixed string, not a reading of
# what happened.
NO_KNOWN_REPAIR_STOP = "no known DelegateDoctor repair matched"

# Backend fidelity is measured on two surfaces - ExecuTorch on the host, and
# ExecuTorch on the device - and reported as one status.
_FIDELITY_SEVERITY = {"OK": 0, "WARNING": 1, "FAIL": 2}


def _worse_fidelity(host_result, device_result) -> tuple:
    """The more serious of the two backend-fidelity readings, and its reason."""
    readings = [
        (getattr(result, "backend_fidelity", "OK") or "OK",
         getattr(result, "backend_fidelity_reason", "") or "")
        for result in (host_result, device_result) if result is not None
    ]
    if not readings:
        return "OK", ""
    return max(readings, key=lambda reading: _FIDELITY_SEVERITY.get(reading[0], 0))


def _nothing_left_message() -> str:
    """Why the loop stopped, in terms of what was actually observed."""
    return NO_KNOWN_REPAIR_STOP


def _ai_status_from(history) -> str:
    """One word for how AI figured in this run, from the one consent decision."""
    return {
        repair_loop.AI_CONSENT_GRANTED: repair_opportunity.AI_ATTEMPTED,
        repair_loop.AI_CONSENT_DECLINED: repair_opportunity.AI_DECLINED,
        repair_loop.AI_CONSENT_UNAVAILABLE: repair_opportunity.AI_UNAVAILABLE,
    }.get(history.ai_consent, repair_opportunity.AI_NOT_REQUESTED)


class _RepairMachinery:
    """Drives the iterative repair loop, holding the three programs apart.

    Every device mechanism it uses already existed; what is new is that they
    now run once per hotspot instead of once per run, and that the three
    programs `repair_loop` describes are kept genuinely distinct:

        original   never mutated, and the reference every candidate is
                   verified against
        current    original plus every accepted repair
        candidate  a deep copy of current with one proposed repair

    The candidate becomes current only after passing every gate. Nothing here
    relaxes a gate; the loop just applies them more often.
    """

    def __init__(self, outcome, run_dir, model_spec, args, device_info,
                 bench_runner, etdump_runner, device_input_paths,
                 original_pte_path, original_host_output, eager_tensor,
                 profile_iterations, threads, warmup_iterations,
                 measured_iterations, repetitions, verbose, suppressed_noise,
                 interactive, ai_repair, prompt, emit, note):
        self.outcome = outcome
        self.run_dir = run_dir
        self.model_spec = model_spec
        self.args = args
        self.device_info = device_info
        self.bench_runner = bench_runner
        self.etdump_runner = etdump_runner
        self.device_input_paths = device_input_paths
        self.original_pte_path = original_pte_path
        self.original_host_output = original_host_output
        self.eager_tensor = eager_tensor
        self.profile_iterations = profile_iterations
        self.threads = threads
        self.warmup_iterations = warmup_iterations
        self.measured_iterations = measured_iterations
        self.repetitions = repetitions
        self.verbose = verbose
        self.suppressed_noise = suppressed_noise
        self.interactive = interactive
        self.ai_repair = ai_repair
        self.prompt = prompt
        self.emit = emit
        self.note = note

        self.current_program = None
        self.current_export = None
        self.current_profile = None
        self.current_delegation = None

        self.last_host_verification = None
        self.last_device_verification = None
        self.last_decision = None
        # The step benchmark of the most recent accepted repair. With exactly
        # one accepted repair this *is* the original-vs-final comparison, so
        # re-measuring would only produce a second number to disagree with.
        self.accepted_step_benchmark = None
        self.accepted_count = 0

        self.numbering = repair_loop.CandidateNumbering()
        self._provider = None
        self._provider_resolved = False
        # None until AI first comes up; then True or False for the whole run.
        self._ai_decision = None
        # Candidates this run's exploration produced, still to be tried. One
        # exploration can propose several; each becomes an ordinary candidate.
        self._pending_candidates = []
        self._explored = False
        # Hotspots whose story is over for this run, by stable identity.
        self._finished = set()
        # Catalog rules already attempted, keyed by (graph state, rule id). A
        # deterministic rule on an unchanged graph produces an identical
        # candidate, so attempting it twice can only waste a device benchmark
        # and add a duplicate row to the report.
        self._attempted_rules = set()
        # Candidate graphs already evaluated this run, as a second net: two
        # different rules can converge on the same rewritten graph, and that is
        # still one candidate worth measuring once.
        self._evaluated_candidates = set()
        self._step = 0

    # --- the loop ----------------------------------------------------------

    def run(self, current_program, current_export, current_profile,
            current_delegation) -> repair_loop.RepairHistory:
        self.current_program = current_program
        self.current_export = current_export
        self.current_profile = current_profile
        self.current_delegation = current_delegation

        history = repair_loop.RepairHistory()
        lookup = repair_loop.catalog_lookup_for(ALL_RULES)

        for iteration in range(1, repair_loop.MAX_REPAIR_ITERATIONS + 1):
            history.iterations = iteration

            hotspots = repair_loop.collect_hotspots(
                self.current_profile, self.current_program, lookup)

            # 1. Known repairs first, every single time. This runs again after
            #    an accepted AI repair, because the re-profile may well have
            #    exposed a hotspot a rule does recognise - and a deterministic
            #    answer should never wait behind a provider request.
            #
            #    Scheduled by RULE, not by site: a rule's apply() rewrites every
            #    site it recognises, so asking it once per hotspot produced the
            #    same candidate over and over.
            fingerprint = repair_loop.graph_fingerprint(self.current_program)
            match = repair_loop.next_catalog_match(
                hotspots, self._attempted_rules, fingerprint)

            if match is not None:
                self._step += 1
                self.note(repair_loop.format_match_header(
                    self._step, match, verbose=self.verbose))
                attempt = self._consider(match, iteration)
                history.record(attempt)
                # This rule is finished for this graph state whatever happened.
                # If it was accepted the graph changes, so the new fingerprint
                # lets it run again on genuinely new work; if it was rejected
                # the graph is unchanged and retrying would repeat the identical
                # candidate against the identical gates.
                self._attempted_rules.add(
                    repair_loop.attempt_key(fingerprint, match.rule_id))
                self._finished.update(match.hotspot_ids)
                self.note(repair_loop.format_attempt_result(attempt))
                if not attempt.accepted:
                    continue
                if not self._reprofile():
                    history.stop_reason = "device profiling became unavailable"
                    break
                continue

            # 2. No known repair applies. The remaining question is about the
            #    model as a whole: is enough of it still running outside the
            #    delegate to be worth a bounded investigation?
            decision = model_exploration.assess(self.current_profile)
            if not decision.eligible:
                history.stop_reason = decision.reason or _nothing_left_message()
                if self.current_profile is not None and decision.reason:
                    self.note(model_exploration.format_skipped(decision))
                break

            if not self._authorize_ai(decision, history):
                break

            accepted = self._explore_model(iteration, history)
            if accepted is None:
                # Every candidate this exploration produced has been tried.
                history.stop_reason = "AI exploration produced no further repair"
                break
            if not self._reprofile():
                history.stop_reason = "device profiling became unavailable"
                break
        else:
            history.stop_reason = (
                f"the {repair_loop.MAX_REPAIR_ITERATIONS}-iteration safety "
                f"cap was reached")

        return history

    def _authorize_ai(self, decision, history) -> bool:
        """Ask once, for the whole run. Returns False when exploration stops.

        The question is about the model: "enough of this still runs outside
        the delegate to be worth investigating - may I look?". Asked once, and
        the answer stands for the run however many candidates follow.

        Declining or being unavailable is never a failure: the run finishes
        with the known repairs it already made.
        """
        from .agent import consent

        if self._ai_decision is not None:
            return self._ai_decision

        if not self.ai_repair:
            # The default product. Not a refusal and not a missing credential:
            # experimental repair was simply not asked for, and saying more
            # than that would make an opt-in feature look like a failed step.
            self._ai_decision = False
            history.ai_consent = repair_loop.AI_CONSENT_NOT_ENABLED
            history.stop_reason = NO_KNOWN_REPAIR_STOP
            return False

        self.outcome.ai_repair_requested = True
        provider = self._resolve_provider()
        if provider is None:
            self._ai_decision = False
            history.ai_consent = repair_loop.AI_CONSENT_UNAVAILABLE
            history.stop_reason = "no configured AI provider or credential"
            self.note("\nAI exploration          unavailable")
            self.note("Reason                  no configured AI provider "
                      "or credential")
            return False

        configuration = getattr(provider, "configuration", None)
        label = (configuration.describe() if configuration is not None else "")
        self.note(model_exploration.format_enabled_screen(decision, label))

        # `--ai-repair` *is* the authorization. Asking again would be a second
        # confirmation for a decision already made explicitly on the command
        # line, and would make the flag unusable non-interactively. The notice
        # below is informational: it states what leaves the machine before the
        # first request, which is a different obligation from consent.
        self.note(consent.repair_privacy_notice(configuration))
        self._ai_decision = True
        history.ai_consent = repair_loop.AI_CONSENT_GRANTED
        self.outcome.ai_repair_attempted = True
        if configuration is not None:
            self.outcome.ai_provider = configuration.definition.label
            self.outcome.ai_model = configuration.model
        return True

    def _explore_model(self, iteration, history):
        """One bounded investigation of the current model, then its candidates.

        The provider is asked once about the whole graph and may propose
        several transformations. Each proposal becomes an ordinary repair
        candidate and meets exactly the gates a catalog rule does - there is
        no separate AI verification path, and there never was a reason for one.

        Returns True when a candidate was accepted, None when the exploration
        is finished either way.
        """
        from .agent import repair_explorer

        provider = self._resolve_provider()
        if provider is None:                              # pragma: no cover
            return None

        if not self._explored:
            self._explored = True
            self.note(model_exploration.format_exploration_start())

            context, known = model_exploration.build_model_context(
                self.current_program, self.current_profile,
                self.current_delegation, ALL_RULES,
                android_setup.SUPPORTED_EXECUTORCH_VERSION)
            if not known:
                self.note("    The exported graph has no describable "
                          "computation nodes.")
                return None

            try:
                exploration = repair_explorer.explore(
                    provider=provider,
                    baseline_program=self.current_program,
                    context=context,
                    known_nodes=known,
                    lower=export_model.lower_with_xnnpack,
                    announce=lambda line: None,
                    candidate_id_factory=self.numbering.next,
                )
            except Exception as error:                    # pragma: no cover
                self.note(f"    AI exploration stopped: {type(error).__name__}")
                return None

            from .agent import provider_response

            provider_label = self._provider_label()
            result = exploration.provider_result

            if result is not None and not result.succeeded:
                # The provider did not answer usably. Nothing was proposed, so
                # nothing is counted and no RepairAttempt is invented.
                self.note(provider_response.format_outcome(
                    result, provider_label, verbose=self.verbose))
                self.note(f"Candidates proposed     0")
                history.ai_provider_status = result.reported_status
                history.ai_provider_detail = result.message
                return None

            if exploration.declined:
                # A successful response whose content is "no safe repair".
                # Reported under its own name: the call succeeded, and saying
                # REPAIR_PROPOSALS_RETURNED here would claim proposals that
                # were deliberately not made.
                declined = provider_response.ProviderCompletionResult(
                    provider_response.SUCCESS,
                    message="provider found no safe DSL-expressible repair",
                    diagnostics=dict(result.diagnostics) if result else {})
                declined.status = provider_response.NO_REPAIR_PROPOSED
                self.note(provider_response.format_outcome(
                    declined, provider_label, verbose=self.verbose))
                self.note("Candidates proposed     0")
                history.ai_provider_status = provider_response.NO_REPAIR_PROPOSED
                history.ai_provider_detail = (
                    "provider found no safe DSL-expressible repair")
                return None

            # Only now did real proposals arrive, so only now do they count.
            self.outcome.ai_candidate_count += exploration.candidates_proposed
            history.ai_candidates_proposed += exploration.candidates_proposed
            history.ai_provider_status = (
                provider_response.REPORTED_STATUS[provider_response.SUCCESS]
                if exploration.candidates_proposed else
                provider_response.NO_REPAIR_PROPOSED)
            self.outcome.ai_attempt_summaries += [
                candidate.to_dict() for candidate in exploration.attempts]
            self._pending_candidates = list(exploration.runnable_candidates)
            self._record_unrunnable(exploration, iteration, history)

        if not self._pending_candidates:
            return None

        program, plan = self._pending_candidates.pop(0)
        history.ai_candidates_tested += 1
        self._step += 1
        self.note(f"\n{plan.candidate_id}")
        self.note("    Applying...")

        attempt = repair_loop.RepairAttempt(
            iteration=iteration, hotspot=None,
            source=repair_loop.SOURCE_AI, candidate_id=plan.candidate_id)
        self._evaluate(program, attempt)
        history.record(attempt)
        self.note(repair_loop.format_attempt_result(attempt))
        return True if attempt.accepted else self._explore_model(
            iteration, history)

    def _provider_label(self) -> str:
        configuration = getattr(self._provider, "configuration", None)
        return configuration.describe() if configuration is not None else ""

    def _record_unrunnable(self, exploration, iteration, history) -> None:
        """Proposals that never became a graph are still part of the story."""
        for candidate in exploration.attempts:
            if candidate.outcome == "runnable":
                continue
            self.note(f"\n{candidate.candidate_id}")
            self.note(f"    Result                 "
                      f"{repair_loop.NO_CANDIDATE}")
            if self.verbose and candidate.detail:
                self.note(f"    Detail                 {candidate.detail}")
            history.record(repair_loop.RepairAttempt(
                iteration=iteration, hotspot=None,
                source=repair_loop.SOURCE_AI,
                candidate_id=candidate.candidate_id,
                status=repair_loop.NO_CANDIDATE,
                reason=candidate.outcome))


    def _consider(self, match, iteration) -> repair_loop.RepairAttempt:
        """Produce one candidate for one catalog rule and put it through the gates.

        A catalog repair is applied without asking. It is deterministic, it is
        already understood, and the gates below decide whether it survives -
        there is nothing for a user to weigh in on beforehand that the
        benchmark will not answer better a minute later.
        """
        attempt = repair_loop.RepairAttempt(
            iteration=iteration,
            hotspot=match.primary,
            measured_sites=match.measured_site_count,
            represented_runtime=match.runtime_share,
        )

        self.note(f"    {match.rule_id} found")
        candidate = self._catalog_candidate(match, attempt)
        if candidate is None:
            return attempt

        self._evaluate(candidate, attempt)
        return attempt

    # --- producing a candidate ---------------------------------------------

    def _catalog_candidate(self, match, attempt):
        """A deep copy of the current program with the matching rule applied.

        The rule is applied to every site it recognises. That is what the rules
        have always done - `apply()` walks the whole graph - and it is why one
        attempt per rule is the honest unit: there is no such thing as applying
        a catalog rule to only one of its sites.
        """
        rule = next((candidate_rule for candidate_rule in ALL_RULES
                     if candidate_rule.RULE_ID == match.rule_id), None)
        if rule is None:
            attempt.status = repair_loop.NOT_APPLICABLE
            attempt.reason = "the matching rule is no longer in the catalog"
            return None

        candidate = copy.deepcopy(self.current_program)
        found = rule.detect(candidate)
        if not found.applies:
            # The kernel name matched but the pattern is not in *this* graph -
            # an earlier repair may already have removed it.
            attempt.status = repair_loop.NOT_APPLICABLE
            attempt.reason = f"{rule.RULE_ID} no longer matches this graph"
            return None

        self.note(f"    Applying to {len(found.detections)} matching site"
                  f"{'' if len(found.detections) == 1 else 's'}...")
        sites = rule.apply(candidate)
        attempt.source = repair_loop.SOURCE_CATALOG
        attempt.repair_id = rule.RULE_ID
        attempt.matching_sites = sites
        self.emit(reporting.format_detection(rule, found))
        self.emit(reporting.format_repair(rule, sites))

        fingerprint = repair_loop.graph_fingerprint(candidate)
        if fingerprint and fingerprint in self._evaluated_candidates:
            attempt.status = repair_loop.SKIPPED
            attempt.reason = ("this exact candidate graph was already measured "
                              "in this run")
            return None
        if fingerprint:
            self._evaluated_candidates.add(fingerprint)
        return candidate




    def _kernel_for(self, hotspot):
        """The profiled kernel this hotspot came from, for the AI context."""
        for kernel in self.current_profile.portable_kernels:
            if kernel.name == hotspot.kernel_name:
                return kernel
        return None

    def _resolve_provider(self):
        """Build the provider at most once, and only when a hotspot needs it.

        Lazy on purpose: a deterministic run with no AI configured must never
        be delayed, warned at, or failed by a credential lookup it did not
        need. A provider that cannot be built is simply "AI unavailable".
        """
        if self._provider_resolved:
            return self._provider
        self._provider_resolved = True

        if not self.ai_repair:
            return None
        try:
            from .agent.client import build_provider

            self._provider = build_provider(allow_ai=True)
        except Exception as error:
            self.note(f"    AI repair unavailable: "
                      f"{str(error).splitlines()[0] if str(error) else type(error).__name__}")
            self._provider = None
        return self._provider

    # --- the gates ----------------------------------------------------------

    def _evaluate(self, candidate_program, attempt) -> bool:
        """Lower, verify against the ORIGINAL, benchmark against CURRENT.

        Correctness looks all the way back to the original model, so cumulative
        drift across several accepted repairs cannot pass unnoticed.
        Performance looks only one step back, because the question is whether
        this repair improves what is already accepted.
        """
        step_dir = os.path.join(self.run_dir, f"step_{self._step:02d}")
        os.makedirs(step_dir, exist_ok=True)

        try:
            candidate_export = export_model.lower_and_write(
                candidate_program, os.path.join(step_dir, "model.pte"))
        except Exception as error:
            attempt.status = repair_loop.REJECTED
            attempt.reason = f"lowering failed: {type(error).__name__}"
            return False
        export_model.save_readable_graphs(candidate_export, step_dir)

        with console_noise.filter_native_stderr(verbose=self.verbose,
                                                suppressed=self.suppressed_noise):
            candidate_host_output = export_model.run_on_host(
                candidate_export.pte_path, self.args)[0]

        host_result = verify_repair(
            original_output=self.original_host_output,
            repaired_output=candidate_host_output,
            eager_output=self.eager_tensor,
            argmax_dim=self.model_spec.argmax_dim,
        )
        attempt.host_verification_passed = host_result.passed
        attempt.backend_fidelity = host_result.backend_fidelity
        attempt.backend_fidelity_reason = host_result.backend_fidelity_reason
        self.last_host_verification = host_result

        # A semantically invalid candidate is finished here. Running the device
        # verification and then a full p50 benchmark on it would spend minutes
        # of device time to measure how fast a wrong answer arrives.
        if not host_result.passed:
            self._save_verification(step_dir, host_result, None)
            return self._reject_early(attempt, host_verification_passed=False)
        if not host_result.backend_fidelity_acceptable:
            self._save_verification(step_dir, host_result, None)
            return self._reject_early(attempt, backend_fidelity_acceptable=False)

        # Device correctness compares the candidate against what the ORIGINAL
        # program produced on the target, for the same cumulative reason.
        try:
            device_result = device_verification.run_device_verification(
                before_pte_path=self.original_pte_path,
                after_pte_path=candidate_export.pte_path,
                input_paths=self.device_input_paths,
                bench_runner_path=self.bench_runner,
                original_host_output=self.original_host_output,
                repaired_host_output=candidate_host_output,
                output_dir=step_dir,
                serial=self.device_info.serial,
                argmax_dim=self.model_spec.argmax_dim,
                threads=self.threads,
            )
        except device_verification.DeviceVerificationError as error:
            device_result = device_verification.DeviceVerificationResult(
                passed=False, failure_reasons=[str(error)], error=str(error))
        attempt.device_verification_passed = device_result.passed
        # The worst of the two, so a warning on either surface is visible and a
        # regression on either is what the reader is told about.
        attempt.backend_fidelity, attempt.backend_fidelity_reason = \
            _worse_fidelity(host_result, device_result)
        self.last_device_verification = device_result

        self._save_verification(step_dir, host_result, device_result)
        self.emit(reporting.format_verification(host_result, device_result))

        # Same reasoning as above: a candidate that changed what the device
        # computes, or that made the backend agree with PyTorch materially
        # less well, does not earn a benchmark.
        if not device_result.passed:
            return self._reject_early(attempt, device_verification_passed=False)
        if not device_result.backend_fidelity_acceptable:
            return self._reject_early(attempt, backend_fidelity_acceptable=False)

        step_benchmark = benchmarking.benchmark_before_after(
            before_pte_path=self.current_export.pte_path,
            after_pte_path=candidate_export.pte_path,
            input_paths=self.device_input_paths,
            bench_runner_path=self.bench_runner,
            device_info=self.device_info,
            warmup_iterations=self.warmup_iterations,
            measured_iterations=self.measured_iterations,
            repetitions=self.repetitions,
            threads=self.threads,
        )
        attempt.before_latency_ms = step_benchmark.before.p50_ms
        attempt.after_latency_ms = step_benchmark.after.p50_ms
        reporting.write_json(step_benchmark.to_dict(),
                             os.path.join(step_dir, "benchmark.json"))

        decision = decide_repair(
            host_verification_passed=host_result.passed,
            device_verification_passed=device_result.passed,
            before_latency_ms=step_benchmark.before.p50_ms,
            after_latency_ms=step_benchmark.after.p50_ms,
            backend_fidelity_acceptable=device_result.backend_fidelity_acceptable,
        )
        self.last_decision = decision

        if not decision.accepted:
            attempt.status = repair_loop.REJECTED
            attempt.reason = getattr(decision, "headline", "") or "rejected"
            return False

        attempt.status = repair_loop.ACCEPTED
        self.current_program = candidate_program
        self.current_export = candidate_export
        self.accepted_step_benchmark = step_benchmark
        self.accepted_count += 1
        self.outcome.record(result_module.VERIFICATION, result_module.PASS)
        self.outcome.record(result_module.BENCHMARK, result_module.PASS)
        return True

    def _save_verification(self, step_dir, host_result, device_result) -> None:
        """Persist what was measured, including for a short-circuited reject.

        A candidate rejected before the device work still produced a real host
        measurement, and that measurement is the evidence for the rejection.
        """
        reporting.write_json(
            {"host": host_result.to_dict(),
             "device": device_result.to_dict() if device_result else None},
            os.path.join(step_dir, "verification.json"))

    def _reject_early(self, attempt, **failed_gate) -> bool:
        """Reject without benchmarking, and say so in the usual vocabulary.

        The decision still comes from `decide_repair`, so a correctness
        rejection reads identically whether or not a benchmark happened to run
        first. The latencies are zero because nothing was measured - which is
        the point - and `decide_repair` reaches its correctness branch before it
        looks at them.
        """
        gates = {"host_verification_passed": True,
                 "device_verification_passed": True,
                 "backend_fidelity_acceptable": True}
        gates.update(failed_gate)

        decision = decide_repair(before_latency_ms=0.0, after_latency_ms=0.0,
                                 **gates)
        self.last_decision = decision
        attempt.status = repair_loop.REJECTED
        attempt.reason = decision.headline
        return False

    # --- measuring the new state --------------------------------------------

    def _reprofile(self) -> bool:
        """Re-measure the accepted program. The next ranking comes from here."""
        self.note("\nRe-profiling...")
        try:
            self.current_profile = profiling.profile_model(
                pte_path=self.current_export.pte_path,
                input_path=self.device_input_paths[0],
                etdump_runner_path=self.etdump_runner,
                output_etdump_path=os.path.join(
                    self.run_dir, f"step_{self._step:02d}", "trace.etdump"),
                label=f"step{self._step}",
                serial=self.device_info.serial,
                iterations=self.profile_iterations,
                threads=self.threads,
            )
        except Exception as error:
            self.note(f"(re-profiling failed: {type(error).__name__})")
            return False

        self.current_delegation = delegation.analyze_delegation(
            self.current_export.edge_program_manager)
        return True

    def benchmark_original_against_final(self):
        """The original-vs-final comparison, measured rather than derived.

        With one accepted repair the step benchmark already compared exactly
        those two programs, so it is reused. With several, the step benchmarks
        each compared a different pair and cannot be multiplied together: each
        one interleaved its own repetitions to cancel thermal drift, and
        chaining them would accumulate what each individually removed. So the
        headline gets its own interleaved measurement.
        """
        if self.current_export is None or self.accepted_count == 0:
            return None
        if self.accepted_count == 1:
            return self.accepted_step_benchmark
        try:
            return benchmarking.benchmark_before_after(
                before_pte_path=self.original_pte_path,
                after_pte_path=self.current_export.pte_path,
                input_paths=self.device_input_paths,
                bench_runner_path=self.bench_runner,
                device_info=self.device_info,
                warmup_iterations=self.warmup_iterations,
                measured_iterations=self.measured_iterations,
                repetitions=self.repetitions,
                threads=self.threads,
            )
        except Exception as error:                      # pragma: no cover
            self.note(f"(final benchmark failed: {type(error).__name__})")
            return None
