"""The concise console output.

Offline: no adb, device, NDK, network or runner binaries. These exercise the
formatters directly with small fake result objects.
"""

import pytest

from delegate_doctor import reporting


# --- concise console output ------------------------------------------------

class FakeMetrics:
    def __init__(self, max_abs=1.86e-08):
        self.max_absolute_error = max_abs
        self.mean_absolute_error = 3.2e-09
        self.mean_squared_error = 2.0e-17
        self.max_relative_error = 4.2e-07


class FakeVerification:
    def __init__(self, passed=True, reasons=None):
        self.passed = passed
        self.repaired_vs_original = FakeMetrics()
        self.repaired_vs_eager = FakeMetrics()
        self.original_device_vs_host = FakeMetrics()
        self.repaired_device_vs_host = FakeMetrics()
        self.argmax_agreement = 1.0
        self.failure_reasons = reasons or []
        self.error = ""

    @property
    def status_text(self):
        return "PASS" if self.passed else "FAIL"


def test_header_names_the_architecture_and_device():
    text = reporting.format_header(
        "PSPNet / MobileNetV2", "1x3x256x256", "RMX2030 · arm64-v8a · Android 10")
    assert "PSPNet / MobileNetV2" in text
    assert "1x3x256x256" in text
    assert "RMX2030" in text
    assert "U-Net" not in text


def test_successful_verification_is_two_lines_without_metric_dumps():
    text = reporting.format_verification(FakeVerification(), FakeVerification())
    assert "Host:" in text and "Android:" in text
    assert "argmax 100.00%" in text
    # the long per-metric breakdown must not appear on a clean run
    assert "mean abs" not in text
    assert "max rel" not in text
    assert len([line for line in text.splitlines() if line.strip()]) <= 4


def test_failed_verification_prints_full_diagnostics():
    """Conciseness applies to successes; failures must stay debuggable."""
    failing = FakeVerification(passed=False, reasons=["max abs error above tolerance"])
    text = reporting.format_verification(FakeVerification(), failing)
    assert "FAIL" in text
    assert "tolerance:" in text
    assert "mean abs" in text          # full metrics now shown
    assert "max rel" in text
    assert "max abs error above tolerance" in text


def test_detection_output_is_compact():
    class FakeDetection:
        node_name = "softmax"

        def explain(self):
            return ("softmax: softmax(dim=1) on [1, 21, 256, 256] · rank 4 · "
                    "last dim 3\n  access: 65,536 vectors x 21 classes, "
                    "stride 65,536")

    class FakeResult:
        detections = [FakeDetection()]
        skipped = []

    class FakeRule:
        RULE_ID = "DD-001"
        RULE_TITLE = "non-last-dimension softmax"

    text = reporting.format_detection(FakeRule(), FakeResult())
    assert "softmax(dim=1)" in text
    assert "[1, 21, 256, 256]" in text
    assert "65,536 vectors x 21 classes" in text
    # the old multi-line prose explanation is gone
    assert "XNNPACK requires softmax to operate on the" not in text
    assert len([line for line in text.splitlines() if line.strip()]) <= 4


def test_benchmark_output_is_a_compact_table():
    class Stats:
        def __init__(self, p50, p95, mean):
            self.p50_ms, self.p95_ms, self.mean_ms = p50, p95, mean
            self.p99_ms, self.sample_count = p95 + 5, 450
            self.stdev_ms, self.min_ms, self.max_ms = 1.0, p50 - 5, p95 + 9
            self.throughput_per_second = 1000.0 / mean

    class FakeBenchmark:
        before = Stats(242.69, 284.43, 250.31)
        after = Stats(65.53, 73.40, 68.22)
        threads, measured_iterations, repetitions = 4, 150, 3
        device_is_emulator = False
        p50_speedup = 242.69 / 65.53

    text = reporting.format_benchmark(FakeBenchmark())
    assert "p50" in text and "p95" in text and "mean" in text
    assert "3.70x speedup" in text
    assert "73.0% lower p50" in text
    assert len([line for line in text.splitlines() if line.strip()]) <= 9


def test_emulator_benchmark_stays_labelled():
    """Compressing output must not lose the emulator caveat."""
    class Stats:
        p50_ms = p95_ms = mean_ms = 10.0
        p99_ms = 11.0
        sample_count = 450

    class FakeBenchmark:
        before = after = Stats()
        threads, measured_iterations, repetitions = 4, 150, 3
        device_is_emulator = True
        p50_speedup = 1.0

    assert "emulator" in reporting.format_benchmark(FakeBenchmark())


def test_accepted_decision_names_the_device():
    class FakeDecision:
        accepted = True
        headline = "REPAIR ACCEPTED"
        speedup = 3.70
        message = "..."

    text = reporting.format_decision(FakeDecision(), "RMX2030 · arm64-v8a · Android 10")
    assert "ACCEPTED" in text
    assert "3.70x" in text
    assert "RMX2030" in text


def test_rejected_decision_still_explains_why():
    class FakeDecision:
        accepted = False
        headline = "REPAIR REJECTED - Android numerical verification failed"
        speedup = 2.9
        message = "The tensors produced on the Android device did not verify."

    text = reporting.format_decision(FakeDecision(), "RMX2030")
    assert "REJECTED" in text
    assert "did not verify" in text


# --- the concise terminal summary -------------------------------------------

# Imported under aliases: this module already defines its own FakeVerification
# for the formatter tests above, and shadowing it would silently change them.
from delegate_doctor import result as result_module          # noqa: E402
from tests.test_html_report import (                          # noqa: E402
    FakeDelegation as _FakeDelegation,
    FakeDetection as _FakeDetection,
    accepted_result,
    build,
    healthy_result,
)


def summary_of(outcome) -> str:
    return reporting.format_summary(outcome)


def test_the_summary_is_short():
    """The console default has to fit in a glance, not a scrollback."""
    lines = [line for line in summary_of(accepted_result()).splitlines()
             if line.strip()]
    assert len(lines) <= 12, lines


def test_the_summary_names_the_model_and_the_essentials():
    text = summary_of(accepted_result())
    assert "DelegateDoctor - PSPNet" in text
    for label in ("Result", "Top hotspot", "Runtime delegation", "Latency",
                  "Speedup", "Correctness", "Report"):
        assert label in text, f"summary is missing {label}"


def test_the_summary_reports_the_decision_and_numbers():
    text = summary_of(accepted_result())
    assert "REPAIR ACCEPTED" in text
    assert "242.69 -> 65.53 ms" in text
    assert "3.70x" in text
    assert "34.3% -> 99.4%" in text
    assert "PASS host / PASS device" in text


def test_a_healthy_summary_says_so_without_fake_numbers():
    text = summary_of(healthy_result())
    assert "FULLY DELEGATED" in text
    assert "Portable hotspots       none" in text
    assert "not required" in text
    assert "Speedup" not in text
    assert "Latency" not in text


def test_a_static_summary_does_not_imply_measurement():
    outcome = build(result_module.ANALYSIS_COMPLETE,
                    before_delegation=_FakeDelegation(total=41, portable=1),
                    detections={"DD-001": _FakeDetection()})
    outcome.record(result_module.DEVICE, result_module.UNAVAILABLE, "no target")
    text = summary_of(outcome)
    assert "ANALYSIS COMPLETE" in text
    assert "Operator delegation" in text
    assert "Device                  unavailable" in text
    assert "Runtime profiling       not run" in text
    assert "Repair candidate        DD-001" in text
    assert "Latency" not in text
    assert "Speedup" not in text


def test_a_lowering_failure_summary_names_executorch():
    outcome = build(result_module.EXECUTORCH_LOWERING_UNSUPPORTED)
    outcome.record(result_module.LOWERING, "FAILED",
                   "RuntimeError: edge dialect rejected aten.exotic")
    text = summary_of(outcome)
    assert "EXECUTORCH LOWERING UNSUPPORTED" in text
    assert "lowering failed" in text
    assert "aten.exotic" in text


def test_the_summary_points_at_the_html_report():
    outcome = accepted_result()
    outcome.report_path = "/tmp/run_042/report.html"
    assert "Report                  /tmp/run_042/report.html" in summary_of(outcome)


def test_the_summary_shows_the_optimized_artifact_only_when_there_is_one():
    accepted = accepted_result()
    assert "Optimized model" in summary_of(accepted)
    assert "Optimized model" not in summary_of(healthy_result())


def test_no_terminal_output_contains_emoji():
    for outcome in (accepted_result(), healthy_result()):
        for character in summary_of(outcome):
            code = ord(character)
            assert not (0x1F000 <= code <= 0x1FAFF)
            assert not (0x2600 <= code <= 0x27BF)
