"""Latency benchmarking on the Arm64 target.

Deliberately separate from `profiling.py`:

  * profiling answers "where is time being spent?" and needs the event tracer,
    which adds per-instruction overhead;
  * benchmarking answers "did the repair make the model faster?" and must not
    be perturbed, so it uses the tracer-free runner build.

Timing comes from ExecuTorch's own instrumentation. The stock `executor_runner`
logs a line per iteration:

    Iteration 7 of 220: 1.612 ms

so DelegateDoctor parses those rather than timing the adb round-trip, which
would be dominated by process start-up and USB latency.

The two models are run interleaved (before, after, before, after, ...) across
repetitions. If the device warms up, throttles, or drifts during a long run,
interleaving spreads that effect across both models instead of penalising
whichever ran second.
"""

from __future__ import annotations

import os
import re
import statistics
from dataclasses import dataclass, field
from typing import List

from . import device

# Matches the runner's per-iteration log line, e.g.
#   "Iteration 12 of 220: 1.612000 ms"
ITERATION_LINE_PATTERN = re.compile(r"Iteration (\d+) of (\d+): ([0-9.]+) ms")


@dataclass
class LatencyStats:
    """Latency distribution for one model, in milliseconds."""

    sample_count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    stdev_ms: float
    min_ms: float
    max_ms: float

    @property
    def throughput_per_second(self) -> float:
        if self.mean_ms <= 0:
            return 0.0
        return 1000.0 / self.mean_ms

    def to_dict(self) -> dict:
        return {
            "sample_count": self.sample_count,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "mean_ms": self.mean_ms,
            "stdev_ms": self.stdev_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "throughput_per_second": self.throughput_per_second,
        }


@dataclass
class BenchmarkResult:
    before: LatencyStats
    after: LatencyStats
    warmup_iterations: int
    measured_iterations: int
    repetitions: int
    threads: int
    device_description: str
    device_is_emulator: bool
    raw_before_ms: List[float] = field(default_factory=list)
    raw_after_ms: List[float] = field(default_factory=list)

    @property
    def p50_speedup(self) -> float:
        if self.after.p50_ms <= 0:
            return 0.0
        return self.before.p50_ms / self.after.p50_ms

    def to_dict(self) -> dict:
        return {
            "device": self.device_description,
            "device_is_emulator": self.device_is_emulator,
            "warmup_iterations_per_repetition": self.warmup_iterations,
            "measured_iterations_per_repetition": self.measured_iterations,
            "repetitions": self.repetitions,
            "threads": self.threads,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "p50_speedup": self.p50_speedup,
        }


def percentile(sorted_values: List[float], percent: float) -> float:
    """Nearest-rank percentile, spelled out so the definition is unambiguous."""
    if not sorted_values:
        return 0.0
    rank = int(round(percent / 100.0 * len(sorted_values)))
    if rank < 1:
        rank = 1
    if rank > len(sorted_values):
        rank = len(sorted_values)
    return sorted_values[rank - 1]


def summarize(latencies_ms: List[float]) -> LatencyStats:
    if not latencies_ms:
        raise ValueError("No latency samples were collected.")
    ordered = sorted(latencies_ms)
    return LatencyStats(
        sample_count=len(ordered),
        p50_ms=percentile(ordered, 50),
        p95_ms=percentile(ordered, 95),
        p99_ms=percentile(ordered, 99),
        mean_ms=statistics.fmean(ordered),
        stdev_ms=statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        min_ms=ordered[0],
        max_ms=ordered[-1],
    )


def run_one_pass(
    remote_pte_name: str,
    remote_input_name: str,
    total_iterations: int,
    threads: int,
) -> List[float]:
    """Run the model once for `total_iterations` and return per-iteration times."""
    device.clear_logcat()
    command = (
        f"cd {device.DEVICE_WORK_DIR} && "
        f"./{device.BENCH_RUNNER_NAME} "
        f"--model_path {remote_pte_name} "
        f"--inputs {remote_input_name} "
        f"--num_executions {total_iterations} "
        f"--cpu_threads {threads} "
        f"--print_output none"
    )
    device.run_on_device(command)

    log_text = device.read_executorch_logcat()
    latencies = [
        float(match.group(3)) for match in ITERATION_LINE_PATTERN.finditer(log_text)
    ]
    if len(latencies) != total_iterations:
        raise device.DeviceError(
            f"Expected {total_iterations} iteration timings in logcat but parsed "
            f"{len(latencies)}. The device log may have been truncated; try "
            f"fewer iterations per repetition."
        )
    return latencies


def benchmark_before_after(
    before_pte_path: str,
    after_pte_path: str,
    input_path: str,
    bench_runner_path: str,
    device_info: device.DeviceInfo,
    warmup_iterations: int = 20,
    measured_iterations: int = 150,
    repetitions: int = 3,
    threads: int = 4,
) -> BenchmarkResult:
    """Benchmark two .pte files under identical conditions.

    Identical between the two models: input file, thread count, runner binary,
    ExecuTorch build, device, and iteration counts. The only difference is the
    program itself.
    """
    device.prepare_work_dir()
    device.push_runner(bench_runner_path)

    # Explicit distinct remote names. The two .pte files usually share a
    # basename (before/model.pte and after/model.pte), so pushing them under
    # their basenames would silently overwrite one with the other and benchmark
    # the same program twice.
    remote_before = "before_model.pte"
    remote_after = "after_model.pte"
    remote_input = "benchmark_input.bin"
    device.push_file(before_pte_path, remote_before)
    device.push_file(after_pte_path, remote_after)
    device.push_file(input_path, remote_input)

    total_iterations = warmup_iterations + measured_iterations
    before_samples: List[float] = []
    after_samples: List[float] = []

    for _ in range(repetitions):
        # Interleaved so device drift affects both models equally.
        before_pass = run_one_pass(remote_before, remote_input, total_iterations, threads)
        after_pass = run_one_pass(remote_after, remote_input, total_iterations, threads)
        # The first iterations include lazy allocation and cache warm-up.
        before_samples.extend(before_pass[warmup_iterations:])
        after_samples.extend(after_pass[warmup_iterations:])

    return BenchmarkResult(
        before=summarize(before_samples),
        after=summarize(after_samples),
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        repetitions=repetitions,
        threads=threads,
        device_description=device_info.describe(),
        device_is_emulator=device_info.is_emulator,
        raw_before_ms=before_samples,
        raw_after_ms=after_samples,
    )
