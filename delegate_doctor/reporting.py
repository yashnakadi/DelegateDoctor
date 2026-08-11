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
from typing import List, Optional


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


def format_hotspots(profile_result, repairable_kernel_names: List[str],
                    limit: int = 3) -> str:
    """Portable kernels ranked by measured cost, most expensive first."""
    text = _heading("HOTSPOTS")
    if not profile_result.portable_kernels:
        text += "\nNone. All measured runtime is inside XNNPACK.\n"
        return text

    text += "\n"
    for position, kernel in enumerate(profile_result.portable_kernels[:limit], start=1):
        repair = ("DD-001 available" if kernel.name in repairable_kernel_names
                  else "no repair rule")
        text += (
            f"{position}. {kernel.operator_name} · {kernel.total_ms:.1f} ms · "
            f"{_percent(kernel.runtime_fraction)} runtime · "
            f"x{kernel.call_count} · {repair}\n"
        )
    remaining = len(profile_result.portable_kernels) - limit
    if remaining > 0:
        text += f"   (+{remaining} smaller fallback(s))\n"
    return text


def format_detection(detection_result) -> str:
    """What DD-001 found. One line per site."""
    text = _heading("DD-001  non-last-dimension softmax")
    if not detection_result.detections:
        if not detection_result.skipped:
            text += "\nNo softmax operations found in this graph.\n"
        else:
            text += "\nNot applicable:\n"
            for skipped in detection_result.skipped:
                text += f"  {skipped.node_name}: {skipped.reason}\n"
        return text

    text += "\n"
    for detection in detection_result.detections:
        text += (
            f"{detection.node_name}: softmax(dim={detection.softmax_dim}) on "
            f"{list(detection.input_shape)} · rank {detection.tensor_rank} · "
            f"last dim {detection.last_dim}\n"
            f"  access: {detection.vector_count:,} vectors x "
            f"{detection.vector_length} classes, stride "
            f"{detection.element_stride:,}\n"
        )
    return text


def format_repair(rewrite_description: str, repaired_count: int) -> str:
    return (
        f"\nRepair: view -> permute -> softmax(dim=-1) -> permute -> view"
        f"  ({repaired_count} site(s))\n"
    )


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
        "rule": "DD-001",
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
