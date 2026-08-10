"""Terminal output and machine-readable results.

Plain text on purpose - no terminal UI library. Each function formats one
section and returns a string, so the CLI can print sections as they complete
(device runs take a while) and still save the whole thing to report.txt at the
end.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

SECTION_RULE = "-" * 40


def _heading(title: str) -> str:
    return f"\n{title}\n{SECTION_RULE}\n"


def _percent(fraction: float) -> str:
    return f"{100 * fraction:.1f}%"


def format_header(model_name: str, description: str, target_description: str) -> str:
    lines = [
        "DelegateDoctor",
        "",
        f"Model: {model_name}",
    ]
    if description:
        lines.append(f"Description: {description}")
    lines.append("Backend: ExecuTorch + XNNPACK")
    lines.append(f"Target: {target_description}")
    return "\n".join(lines)


def format_analysis(
    delegation_report,
    profile_result,
) -> str:
    """Operator-count delegation next to runtime-weighted delegation.

    Showing them together is the point of the tool: when they disagree, the
    operator count is the misleading one.
    """
    operator_fraction = delegation_report.operator_delegation_fraction
    runtime_fraction = profile_result.runtime_delegation_fraction

    text = _heading("ANALYSIS")
    text += (
        f"Graph operators:             {delegation_report.total_ops}\n"
        f"Delegated operators:         {delegation_report.delegated_op_total}\n"
        f"Portable operators:          {delegation_report.portable_op_total}\n"
        f"XNNPACK delegate blobs:      {delegation_report.delegate_blob_count}\n"
        f"\n"
        f"Operator-count delegation:   {_percent(operator_fraction)}\n"
        f"Runtime-weighted delegation: {_percent(runtime_fraction)}\n"
        f"\n"
        f"Measured on device: {profile_result.method_execute_ms:.3f} ms per inference\n"
        f"  inside XNNPACK:   {profile_result.delegated_ms:.3f} ms "
        f"({profile_result.delegate_call_count} delegate call(s))\n"
        f"  portable kernels: {profile_result.portable_ms:.3f} ms "
        f"({profile_result.operator_call_count} operator call(s))\n"
    )

    if profile_result.accounting_warning:
        text += f"\nNOTE: {profile_result.accounting_warning}\n"

    # Flag the situation the tool exists to find.
    if operator_fraction - runtime_fraction > 0.10:
        text += (
            "\nWARNING:\n"
            "A small number of fallback operations\n"
            "dominate model runtime.\n"
        )
    return text


def format_hotspots(profile_result, repairable_kernel_names: List[str]) -> str:
    """Portable kernels ranked by measured cost, most expensive first."""
    text = _heading("FALLBACK HOTSPOTS")

    if not profile_result.portable_kernels:
        text += "None. All measured runtime is inside XNNPACK.\n"
        return text

    for position, kernel in enumerate(profile_result.portable_kernels, start=1):
        has_repair = kernel.name in repairable_kernel_names
        text += (
            f"\n{position}. {kernel.operator_name}\n"
            f"\n"
            f"Portable runtime:            {kernel.total_ms:.3f} ms "
            f"({kernel.call_count} call(s))\n"
            f"Runtime impact:              {_percent(kernel.runtime_fraction)}\n"
        )
        if has_repair:
            text += (
                f"\nKnown repair:\n"
                f"DD-001 - non-last-dimension softmax\n"
                f"\nRepair available: YES\n"
            )
        else:
            text += "\nRepair available: NO (no rule for this operator yet)\n"
    return text


def format_detection(detection_result) -> str:
    """What DD-001 found, and what it deliberately left alone."""
    text = _heading("DD-001 DETECTION")

    if not detection_result.detections and not detection_result.skipped:
        text += "No softmax operations found in this graph.\n"
        return text

    for detection in detection_result.detections:
        text += detection.explain() + "\n"
        text += (
            f"\nAccess pattern:\n"
            f"{detection.vector_count} softmax vectors of {detection.vector_length} "
            f"elements,\n"
            f"{detection.element_stride} elements apart in memory.\n\n"
        )
    for skipped in detection_result.skipped:
        text += f"Skipped {skipped.node_name}: {skipped.reason}\n"
    return text


def format_repair(rewrite_description: str, repaired_count: int) -> str:
    text = _heading("REPAIR")
    text += f"Applying DD-001 to {repaired_count} site(s)...\n\n"
    text += rewrite_description + "\n"
    text += "\nRe-exporting with XNNPACK...\n"
    return text


def format_delegation_change(
    before_delegation,
    after_delegation,
    before_profile,
    after_profile,
) -> str:
    text = _heading("DELEGATION AFTER REPAIR")
    text += (
        f"                             BEFORE      AFTER\n"
        f"\n"
        f"Portable operators           {before_delegation.portable_op_total:>6}      "
        f"{after_delegation.portable_op_total:>6}\n"
        f"Operator-count delegation    {_percent(before_delegation.operator_delegation_fraction):>6}      "
        f"{_percent(after_delegation.operator_delegation_fraction):>6}\n"
        f"Runtime-weighted delegation  {_percent(before_profile.runtime_delegation_fraction):>6}      "
        f"{_percent(after_profile.runtime_delegation_fraction):>6}\n"
    )
    return text


def format_verification(verification_result) -> str:
    text = _heading("VERIFICATION")
    metrics = verification_result.repaired_vs_original
    text += (
        f"Repaired vs original ExecuTorch output:\n"
        f"  Max absolute error:        {metrics.max_absolute_error:.3e}\n"
        f"  Mean absolute error:       {metrics.mean_absolute_error:.3e}\n"
        f"  Mean squared error:        {metrics.mean_squared_error:.3e}\n"
        f"  Max relative error:        {metrics.max_relative_error:.3e}\n"
    )
    if verification_result.repaired_vs_eager is not None:
        eager_metrics = verification_result.repaired_vs_eager
        text += (
            f"\nRepaired vs PyTorch eager output:\n"
            f"  Max absolute error:        {eager_metrics.max_absolute_error:.3e}\n"
        )
    if verification_result.argmax_agreement is not None:
        text += (
            f"\nArgmax agreement:            "
            f"{100 * verification_result.argmax_agreement:.4f}%\n"
        )
    for reason in verification_result.failure_reasons:
        text += f"\nFAILURE: {reason}\n"
    text += f"\nNumerical verification: {verification_result.status_text}\n"
    return text


def format_benchmark(benchmark_result) -> str:
    before = benchmark_result.before
    after = benchmark_result.after

    text = _heading("BENCHMARK")
    text += (
        f"Target: {benchmark_result.device_description}\n"
        f"Threads: {benchmark_result.threads}   "
        f"Warmup: {benchmark_result.warmup_iterations}/rep   "
        f"Measured: {benchmark_result.measured_iterations}/rep x "
        f"{benchmark_result.repetitions} reps = {before.sample_count} samples\n"
        f"Runner: tracer-free executor_runner (no profiling instrumentation)\n"
        f"\n"
        f"                         BEFORE      AFTER\n"
        f"\n"
        f"p50 latency          {before.p50_ms:>10.3f} ms  {after.p50_ms:>8.3f} ms\n"
        f"p95 latency          {before.p95_ms:>10.3f} ms  {after.p95_ms:>8.3f} ms\n"
        f"p99 latency          {before.p99_ms:>10.3f} ms  {after.p99_ms:>8.3f} ms\n"
        f"mean latency         {before.mean_ms:>10.3f} ms  {after.mean_ms:>8.3f} ms\n"
        f"throughput           {before.throughput_per_second:>10.1f}/s  "
        f"{after.throughput_per_second:>8.1f}/s\n"
        f"\n"
        f"Speedup (p50):       {benchmark_result.p50_speedup:>10.2f}x\n"
    )
    if benchmark_result.device_is_emulator:
        text += (
            "\nNOTE: measured on an Arm64 Android emulator, not a handset. "
            "Arm64 code\nruns natively, but cache sizes, memory bandwidth and "
            "CPU scheduling differ\nfrom a phone. Treat the exact multiplier as "
            "provisional.\n"
        )
    return text


def format_decision(decision) -> str:
    text = _heading("DECISION")
    text += f"{decision.headline}\n\n{decision.message}\n"
    return text


def build_results_json(
    model_name: str,
    device_description: str,
    before_delegation,
    after_delegation,
    before_profile,
    after_profile,
    verification_result,
    benchmark_result,
    decision,
) -> dict:
    """The small machine-readable summary of a run."""
    return {
        "rule": "DD-001",
        "model": model_name,
        "device": device_description,
        "verification_passed": verification_result.passed,
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
