"""Terminal output and machine-readable results.

Plain text on purpose - no terminal UI library. Each function formats one
section and returns a string, so the CLI can print sections as they complete
(device runs take a while) and still save the whole thing to report.txt.

Presentation rule: successful runs are terse, failures are not. A run that
passes every gate should fit on one screen, because the only things a developer
needs from it are what was wrong, what changed, whether it is still correct and
whether it got faster. When a gate fails, the detailed numbers are printed
instead, because that is when they matter.

Nothing here participates in a decision - it only formats. The full metrics are
always written to the run artifacts regardless of what is shown.
"""

from __future__ import annotations

import json
import os


def _heading(title: str) -> str:
    return f"\n{title}"


def _percent(fraction: float) -> str:
    return f"{100 * fraction:.1f}%"


def format_header(model_name: str, input_shape: str, target_description: str) -> str:
    return (
        f"DelegateDoctor\n"
        f"\n"
        f"Model:  {model_name}\n"
        f"Input:  {input_shape}\n"
        f"Device: {target_description}\n"
        f"Backend: ExecuTorch + XNNPACK"
    )


# What the run's AI exploration amounted to, in one phrase. A provider that
# never answered proposed nothing, and saying "1 candidate(s), none accepted"
# about it claimed work that did not happen.
_AI_SUMMARY_TEXT = {
    "PROVIDER_ERROR": "provider call failed",
    "PROVIDER_REFUSED": "provider declined to answer",
    "PROVIDER_EMPTY_RESPONSE": "provider returned no usable response",
    "INVALID_STRUCTURED_RESPONSE": "provider response was not usable",
    "NO_REPAIR_PROPOSED": "completed, no repair proposed",
}


def _ai_exploration_summary(history) -> str:
    status = history.ai_provider_status
    if status in _AI_SUMMARY_TEXT:
        return _AI_SUMMARY_TEXT[status]
    proposed = history.ai_candidates_proposed
    return f"completed, {proposed} proposal(s)"


def format_summary(outcome) -> str:
    """The whole run in a handful of aligned lines.

    This is what the console shows by default. Everything longer lives in
    report.txt and report.html, which are easier to read than a scrollback
    buffer anyway. Only measured values appear here - a stage that did not run
    contributes no line rather than a zero.
    """
    from .result import (DEVICE, EXECUTORCH_LOWERING_UNSUPPORTED, LOWERING,
                         NOT_RUN, PASS, PROFILING)

    lines = [f"\nDelegateDoctor - {outcome.model_name or 'PyTorch Model'}", ""]

    def row(label: str, value: str) -> None:
        lines.append(f"{label:<24}{value}")

    row("Result", outcome.status.replace("_", " "))

    if outcome.stage_status(LOWERING) == "FAILED":
        stage = outcome.stage(LOWERING)
        row("ExecuTorch", "lowering failed")
        if stage and stage.detail:
            row("Cause", stage.detail.splitlines()[0][:70])
    elif outcome.before_profile is not None:
        profile = outcome.before_profile
        hotspots = profile.portable_kernels
        if hotspots:
            top = hotspots[0]
            row("Top hotspot",
                f"{top.operator_name} · {_percent(top.runtime_fraction)} runtime")
        else:
            row("Portable hotspots", "none")

        if outcome.after_profile is not None:
            row("Runtime delegation",
                f"{_percent(profile.runtime_delegation_fraction)} -> "
                f"{_percent(outcome.after_profile.runtime_delegation_fraction)}")
        else:
            row("Runtime delegation",
                _percent(profile.runtime_delegation_fraction))
    else:
        # No profiling: report what static analysis established, and be plain
        # about the device rather than implying a measurement.
        if outcome.before_delegation is not None:
            row("Operator delegation",
                _percent(outcome.before_delegation.operator_delegation_fraction))
        device_stage = outcome.stage(DEVICE)
        if device_stage is not None and device_stage.status != PASS:
            row("Device", device_stage.status.lower())
        if outcome.stage_status(PROFILING) == NOT_RUN:
            row("Runtime profiling", "not run")

    if outcome.benchmark is not None:
        row("Latency", f"{outcome.benchmark.before.p50_ms:.2f} -> "
                       f"{outcome.benchmark.after.p50_ms:.2f} ms")
        row("Speedup", f"{outcome.benchmark.p50_speedup:.2f}x")

    if outcome.host_verification is not None:
        device_text = ("" if outcome.device_verification is None
                       else f" host / {outcome.device_verification.status_text} device")
        row("Correctness",
            outcome.host_verification.status_text + device_text)

    history = outcome.repair_history
    if history is not None and history.any_accepted:
        # Optimization is a sequence now, so the summary counts rather than
        # naming one repair - and says where each came from, because a run
        # mixing catalog and AI repairs is an ordinary outcome.
        row("Accepted repairs", str(history.accepted_count))
        if history.catalog_count:
            row("catalog", str(history.catalog_count))
        if history.ai_count:
            row("AI", str(history.ai_count))
            row("Experimental", "Yes")
            if outcome.ai_provider:
                row("Provider", f"{outcome.ai_provider} · {outcome.ai_model}")
        row("Applied", ", ".join(history.applied_repair_ids))
        if history.rejected_count:
            row("Rejected", str(history.rejected_count))
    elif history is not None and history.rejected_count:
        # A rejected repair is the run's actual finding, so name it and say
        # which gate ended it. The old text called this a "Repair candidate",
        # which described what DelegateDoctor was holding rather than what it
        # had learned - and said it once per duplicate attempt.
        rejected = [attempt for attempt in history.attempts
                    if attempt.status == "REJECTED"]
        row("Repair", ", ".join(sorted({attempt.label for attempt in rejected})))
        last = rejected[-1]
        if last.matching_sites:
            row("Matching sites", str(last.matching_sites))
        if last.reason:
            row("Reason", last.reason)
    elif outcome.repairs_applied:
        row("Repair applied", ", ".join(sorted(outcome.repairs_applied)))
    elif outcome.repair_available:
        row("Repair candidate",
            ", ".join(sorted(rule_id for rule_id, found
                             in outcome.detections.items() if found.applies)))
    elif outcome.status != EXECUTORCH_LOWERING_UNSUPPORTED:
        row("Repair", "not required" if outcome.status in (
            "FULLY_DELEGATED", "NO_REPAIR_REQUIRED") else "none available")
        history = outcome.repair_history
        if history is not None and history.ai_provider_status:
            row("AI exploration", _ai_exploration_summary(history))
            row("Candidates tested", str(history.ai_candidates_tested))
        elif history is not None and \
                history.ai_consent == "not enabled":
            # Experimental AI repair was not opted into. Saying so on every
            # default run would make an opt-in feature look like a step that
            # failed, so nothing is printed at all.
            pass
        elif outcome.ai_repair_attempted:
            row("AI exploration",
                f"{outcome.ai_candidate_count} candidate(s), none accepted")
        elif outcome.ai_repair_requested:
            row("AI exploration", "unavailable")

    if history is not None and history.total_speedup and history.accepted_count > 1:
        row("Total speedup", f"{history.total_speedup:.2f}x")

    lines.append("")
    if outcome.output_pte:
        row("Optimized model", outcome.output_pte)
    row("Report", outcome.report_path or outcome.run_dir)
    return "\n".join(lines)


def format_ai_candidate(exploration) -> str:
    """The one runnable AI candidate, before it faces the ordinary gates."""
    plan = exploration.plan
    text = _heading("AI CANDIDATE") + "\n"
    text += f"{plan.candidate_id}\n"
    if plan.summary:
        text += f"{plan.summary}\n"
    text += (f"{len(plan.operations)} constrained graph operation(s); "
             f"experimental.\n"
             f"It now faces the same host, device and benchmark gates a "
             f"catalog repair does.\n")
    return text


def format_pipeline(stages) -> str:
    """How far the model got, stage by stage.

    Printed on every run, including the ones that stopped early. A stage that
    did not run says so; nothing here ever prints PASS for work not done.
    """
    text = _heading("PIPELINE") + "\n"
    text += "-" * 40 + "\n"
    for stage in stages:
        # Every stage is listed, including the ones that never ran. Silence
        # would leave the reader guessing which of them were skipped.
        text += f"{stage.name:<28}{stage.status}\n"
        if stage.detail:
            text += f"  {stage.detail}\n"
    return text


def format_result(outcome) -> str:
    """The final RESULT block: what this run concluded."""
    text = _heading("RESULT") + "\n"
    text += "-" * 40 + "\n"
    text += f"{outcome.status}\n{outcome.summary}\n"
    return text


def format_lowering_failure(model_name, error, executorch_version) -> str:
    """Export succeeded; ExecuTorch declined the graph. Say precisely that."""
    return (
        f"\nEXECUTORCH LOWERING\nFAILED\n"
        f"\n"
        f"DelegateDoctor captured and inspected the PyTorch ExportedProgram, but\n"
        f"ExecuTorch could not lower the graph into a runnable program. This is a\n"
        f"limitation of the ExecuTorch deployment path for this model, not a\n"
        f"failure of the model or of torch.export.\n"
        f"\n"
        f"  Model:      {model_name}\n"
        f"  ExecuTorch: {executorch_version}\n"
        f"\n"
        f"Cause:\n"
        f"{type(error).__name__}: {str(error)[:900]}\n"
    )


def format_static_analysis(delegation_report, detections, device_status,
                           reason: str, matched=None) -> str:
    """Everything that could be measured without running on the device."""
    text = _heading("DELEGATION") + "\n"
    text += "-" * 40 + "\n"
    text += (
        f"{delegation_report.total_ops} ops · "
        f"{delegation_report.delegated_op_total} delegated · "
        f"{delegation_report.portable_op_total} portable · "
        f"{delegation_report.delegate_blob_count} blob(s)\n"
        f"Operator delegation: "
        f"{_percent(delegation_report.operator_delegation_fraction)}\n"
    )

    text += _heading("ANDROID EXECUTION") + f"\n{device_status}\n"
    if reason:
        text += f"\nReason:\n{reason}\n"
    text += (
        "\nRuntime profiling, verification and benchmarking need the device, so\n"
        "no runtime numbers are reported. Static analysis above is complete.\n"
    )

    if matched:
        text += _heading("REPAIR") + "\n"
        text += f"{', '.join(matched)} pattern(s) matched in this graph.\n"
        text += (
            "Not applied: a repair is only accepted after it verifies and\n"
            "benchmarks faster on the target, and the device was not available.\n"
        )
    text += format_declined_repairs(detections)
    return text


def format_declined_repairs(detections) -> str:
    """Rules that saw their pattern but would not rewrite it.

    Recognising a pattern and being willing to repair it are different things -
    DD-001 declines dynamic shapes, for instance - and the difference is worth
    printing rather than hiding behind "no repair available".
    """
    lines = []
    for rule_id, found in sorted(detections.items()):
        for skipped in getattr(found, "skipped", []):
            lines.append(f"  {rule_id}  {skipped.node_name}: {skipped.reason}")
    if not lines:
        return ""
    return (_heading("CANDIDATE PATTERNS NOT REPAIRED") + "\n"
            + "\n".join(lines) + "\n")


def format_unverifiable(reason: str) -> str:
    """A repair we cannot check is a repair we cannot keep."""
    return (
        f"\nCORRECTNESS VERIFICATION\nUNSUPPORTED\n"
        f"\n"
        f"Reason:\n{reason}\n"
        f"\n"
        f"DelegateDoctor will not accept a repair it cannot verify, so no\n"
        f"repaired artifact was produced. The analysis above still stands.\n"
    )


def format_analysis(delegation_report, profile_result) -> str:
    """Operator-count delegation beside runtime-weighted delegation.

    Showing both on one line is the point of the tool: when they disagree, the
    operator count is the misleading one.
    """
    text = _heading("ANALYSIS")
    text += (
        f"\n{delegation_report.total_ops} ops · "
        f"{delegation_report.delegated_op_total} delegated · "
        f"{delegation_report.portable_op_total} portable · "
        f"{delegation_report.delegate_blob_count} blob(s)\n"
        f"Delegation: {_percent(delegation_report.operator_delegation_fraction)} ops · "
        f"{_percent(profile_result.runtime_delegation_fraction)} runtime "
        f"({profile_result.method_execute_ms:.1f} ms/inference)\n"
    )
    if profile_result.accounting_warning:
        text += f"NOTE: {profile_result.accounting_warning}\n"
    if (delegation_report.operator_delegation_fraction
            - profile_result.runtime_delegation_fraction) > 0.10:
        text += "WARNING: a few fallback ops dominate runtime.\n"
    return text


def format_hotspots(profile_result, repairable_kernels, limit: int = 3) -> str:
    """Portable kernels ranked by measured cost, most expensive first.

    `repairable_kernels` maps a kernel name to the rule id that repairs it, so
    the hotspot list says which rule applies rather than just "available".
    """
    text = _heading("HOTSPOTS")
    if not profile_result.portable_kernels:
        text += "\nNone. All measured runtime is inside XNNPACK.\n"
        return text

    text += "\n"
    for position, kernel in enumerate(profile_result.portable_kernels[:limit], start=1):
        rule_id = repairable_kernels.get(kernel.name)
        repair = f"{rule_id} available" if rule_id else "no repair rule"
        text += (
            f"{position}. {kernel.operator_name} · {kernel.total_ms:.1f} ms · "
            f"{_percent(kernel.runtime_fraction)} runtime · "
            f"x{kernel.call_count} · {repair}\n"
        )
    remaining = len(profile_result.portable_kernels) - limit
    if remaining > 0:
        text += f"   (+{remaining} smaller fallback(s))\n"
    return text


def format_detection(rule, detection_result) -> str:
    """What one rule found. One line per site."""
    text = _heading(f"{rule.RULE_ID}  {rule.RULE_TITLE}")
    if not detection_result.detections:
        if not detection_result.skipped:
            text += "\nNot found in this graph.\n"
        else:
            text += "\nNot applicable:\n"
            for skipped in detection_result.skipped:
                text += f"  {skipped.node_name}: {skipped.reason}\n"
        return text

    text += "\n"
    # A rule with many identical sites prints a count instead of a wall of lines.
    if len(detection_result.detections) > 3:
        first = detection_result.detections[0]
        text += (f"{len(detection_result.detections)} sites, e.g. {first.explain()}\n")
        return text
    for detection in detection_result.detections:
        text += detection.explain() + "\n"
    return text


def format_no_repair(profile_result) -> str:
    """The model was analysed fine, but no catalog rule matches what it uses.

    This is a successful outcome, not an error: the unrepaired hotspots are the
    raw material for a future repair rule.
    """
    text = _heading("NO REPAIR AVAILABLE")
    if not profile_result.portable_kernels:
        text += ("\nNothing to repair - all measured runtime is already inside "
                 "XNNPACK.\n")
        return text
    text += "\nNo verified repair matches this model. Unrepaired fallbacks:\n"
    for kernel in profile_result.portable_kernels[:5]:
        text += (f"  {kernel.operator_name} · {kernel.total_ms:.1f} ms · "
                 f"{_percent(kernel.runtime_fraction)} runtime · no known repair\n")
    text += ("\nThese are candidates for a future repair rule. The model was "
             "analysed successfully;\nno repaired artifact was produced.\n")
    return text


def format_repair(rule, repaired_count: int) -> str:
    return f"Repair: {rule.describe_rewrite()}  ({repaired_count} site(s))\n"


def format_delegation_change(before_delegation, after_delegation,
                             before_profile, after_profile) -> str:
    text = _heading("AFTER REPAIR")
    text += (
        f"\nPortable ops:       {before_delegation.portable_op_total} -> "
        f"{after_delegation.portable_op_total}\n"
        f"Op delegation:      "
        f"{_percent(before_delegation.operator_delegation_fraction)} -> "
        f"{_percent(after_delegation.operator_delegation_fraction)}\n"
        f"Runtime delegation: "
        f"{_percent(before_profile.runtime_delegation_fraction)} -> "
        f"{_percent(after_profile.runtime_delegation_fraction)}\n"
    )
    return text


def _detailed_metrics(label: str, metrics) -> str:
    return (
        f"  {label}\n"
        f"    max abs  {metrics.max_absolute_error:.3e}\n"
        f"    mean abs {metrics.mean_absolute_error:.3e}\n"
        f"    mse      {metrics.mean_squared_error:.3e}\n"
        f"    max rel  {metrics.max_relative_error:.3e}\n"
    )


def format_verification(verification_result, device_result=None) -> str:
    """One line per stage when everything passes; full metrics when it does not.

    The gates themselves are untouched - this only decides what is printed.
    """
    from .verification import MAX_ABSOLUTE_ERROR_TOLERANCE

    text = _heading("VERIFY")

    def agreement(value) -> str:
        return f" · argmax {100 * value:.2f}%" if value is not None else ""

    host = verification_result
    text += (
        f"\nHost:    {host.status_text} · max abs "
        f"{host.repaired_vs_original.max_absolute_error:.2e}"
        f"{agreement(host.argmax_agreement)}\n"
    )

    if device_result is None:
        text += "Android: not run\n"
    else:
        device_error = (
            f"{device_result.repaired_vs_original.max_absolute_error:.2e}"
            if device_result.repaired_vs_original is not None else "n/a"
        )
        text += (
            f"Android: {device_result.status_text} · max abs {device_error}"
            f"{agreement(device_result.argmax_agreement)}\n"
        )

    # Backend fidelity is reported separately, and only when it has something
    # to say. It describes the backend, not the repair, so it never turns the
    # Host or Android line above into a FAIL.
    def fidelity_line(label, result, original, repaired) -> str:
        status = getattr(result, "backend_fidelity", "") if result else ""
        if not status or status == "OK":
            return ""
        def error(metrics):
            return f"{metrics.max_absolute_error:.2e}" if metrics else "n/a"
        line = (f"Backend fidelity ({label}): {status} · original "
                f"{error(original)} · repaired {error(repaired)}\n")
        if result.backend_fidelity_reason:
            line += f"  {result.backend_fidelity_reason}\n"
        return line

    text += fidelity_line("host vs PyTorch", verification_result,
                          getattr(verification_result, "original_vs_eager", None),
                          getattr(verification_result, "repaired_vs_eager", None))
    if device_result is not None:
        text += fidelity_line("device vs host", device_result,
                              device_result.original_device_vs_host,
                              device_result.repaired_device_vs_host)

    host_failed = not verification_result.passed
    device_failed = device_result is not None and not device_result.passed

    # Failures get everything; successes stay on two lines.
    if host_failed or device_failed:
        text += f"\ntolerance: {MAX_ABSOLUTE_ERROR_TOLERANCE:.1e}\n"
        if host_failed:
            text += "\nHost detail:\n"
            text += _detailed_metrics("repaired vs original",
                                      verification_result.repaired_vs_original)
            if verification_result.repaired_vs_eager is not None:
                text += _detailed_metrics("repaired vs PyTorch eager",
                                          verification_result.repaired_vs_eager)
            for reason in verification_result.failure_reasons:
                text += f"  FAILURE: {reason}\n"
        if device_failed:
            text += "\nAndroid detail:\n"
            if device_result.error:
                text += f"  ERROR: {device_result.error}\n"
            if device_result.repaired_vs_original is not None:
                text += _detailed_metrics("repaired vs original (device)",
                                          device_result.repaired_vs_original)
                text += _detailed_metrics("device vs host (original)",
                                          device_result.original_device_vs_host)
                text += _detailed_metrics("device vs host (repaired)",
                                          device_result.repaired_device_vs_host)
            for reason in device_result.failure_reasons:
                text += f"  FAILURE: {reason}\n"
    return text


def format_benchmark(benchmark_result) -> str:
    before = benchmark_result.before
    after = benchmark_result.after
    reduction = (
        100 * (before.p50_ms - after.p50_ms) / before.p50_ms if before.p50_ms else 0.0
    )

    text = _heading("BENCHMARK")
    text += (
        f"\n{benchmark_result.threads} threads · "
        f"{benchmark_result.measured_iterations}x{benchmark_result.repetitions} "
        f"iterations ({before.sample_count} samples) · tracer-free\n"
        f"\n"
        f"           before      after\n"
        f"p50    {before.p50_ms:9.2f}  {after.p50_ms:9.2f} ms\n"
        f"p95    {before.p95_ms:9.2f}  {after.p95_ms:9.2f} ms\n"
        f"mean   {before.mean_ms:9.2f}  {after.mean_ms:9.2f} ms\n"
        f"\n"
        f"{benchmark_result.p50_speedup:.2f}x speedup · {reduction:.1f}% lower p50\n"
    )
    if benchmark_result.device_is_emulator:
        text += "NOTE: Arm64 emulator, not a handset - treat the multiplier as provisional.\n"
    return text


def format_decision(decision, device_description: str = "") -> str:
    text = _heading(decision.headline.replace("REPAIR ", ""))
    if decision.accepted:
        target = f" on {device_description}" if device_description else ""
        text += (
            f"\nCorrect and {decision.speedup:.2f}x faster{target}.\n"
        )
    else:
        text += f"\n{decision.message}\n"
    return text


def build_results_json(
    rules_applied: dict,
    model_name: str,
    device_description: str,
    before_delegation,
    after_delegation,
    before_profile,
    after_profile,
    verification_result,
    device_verification_result,
    benchmark_result,
    decision,
) -> dict:
    """The small machine-readable summary of a run."""
    return {
        "rules_applied": rules_applied,
        "model": model_name,
        "device": device_description,
        "verification_passed": (
            verification_result.passed and device_verification_result.passed
        ),
        "host_verification_passed": verification_result.passed,
        "device_verification_passed": device_verification_result.passed,
        "operator_delegation_before": before_delegation.operator_delegation_fraction,
        "operator_delegation_after": after_delegation.operator_delegation_fraction,
        "runtime_delegation_before": before_profile.runtime_delegation_fraction,
        "runtime_delegation_after": after_profile.runtime_delegation_fraction,
        "p50_before_ms": benchmark_result.before.p50_ms,
        "p50_after_ms": benchmark_result.after.p50_ms,
        "speedup": benchmark_result.p50_speedup,
        "decision": decision.outcome,
    }


def write_json(data: dict, path: str) -> str:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=2)
    return path


def write_text(text: str, path: str) -> str:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as text_file:
        text_file.write(text)
    return path
