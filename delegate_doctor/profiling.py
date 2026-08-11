"""Runtime-weighted delegation: where does inference time actually go?

This is DelegateDoctor's central measurement. Operator-count delegation asks
"what fraction of nodes did XNNPACK take?". Runtime-weighted delegation asks
"what fraction of wall time ran inside XNNPACK?". They can disagree wildly: a
model can be 96.8% delegated by operator count while spending 65% of its time
in the handful of operators that were left behind.

How the number is produced
--------------------------
1. Run the model on the Arm device with the event-tracer build of
   `executor_runner`, which writes an ETDump trace.
2. Pull the trace back and read it with `executorch.devtools.Inspector`.
3. Add up the top-level instruction events.

Step 3 needs care, because ETDump events are nested. For one inference:

    DELEGATE_CALL                       <- one whole XNNPACK blob (top level)
        Convolution (NHWC, F32) IGEMM   <- XNNPACK's own internal nodes
        Transpose (ND, X32)
    OPERATOR_CALL                       <- one portable kernel (top level)
        native_call__softmax.out        <- the kernel's own event
    ...
    Method::execute                     <- total for the whole inference

Summing every row would count the same time two or three times. We therefore
total ONLY the `DELEGATE_CALL` and `OPERATOR_CALL` rows, and sanity-check that
their sum is close to `Method::execute`.

    runtime-weighted delegation = sum(DELEGATE_CALL) / (sum(DELEGATE_CALL) + sum(OPERATOR_CALL))

The nested rows are still useful: the `native_call_*` rows tell us *which*
portable kernel burned the time, which is what turns a number into a hotspot.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from . import device

# Event names ExecuTorch uses for the two kinds of top-level instruction.
DELEGATE_CALL_EVENT = "DELEGATE_CALL"
OPERATOR_CALL_EVENT = "OPERATOR_CALL"
METHOD_EXECUTE_EVENT = "Method::execute"

# Portable kernel events are named 'native_call_<operator>'.
PORTABLE_KERNEL_PREFIX = "native_call_"

# If the top-level rows and Method::execute disagree by more than this, the
# accounting assumption above no longer holds and we should say so rather than
# quietly report a wrong percentage.
ACCOUNTING_TOLERANCE_FRACTION = 0.05


@dataclass
class PortableKernel:
    """One portable (non-delegated) kernel and what it cost."""

    name: str            # e.g. 'native_call__softmax.out'
    total_ms: float
    call_count: int
    runtime_fraction: float  # share of total inference time, 0.0 - 1.0

    @property
    def operator_name(self) -> str:
        """'native_call__softmax.out' -> '_softmax.out'."""
        if self.name.startswith(PORTABLE_KERNEL_PREFIX):
            return self.name[len(PORTABLE_KERNEL_PREFIX):]
        return self.name


@dataclass
class ProfileResult:
    """Runtime breakdown of one model on the Arm target."""

    method_execute_ms: float
    delegated_ms: float
    portable_ms: float
    delegate_call_count: int
    operator_call_count: int
    portable_kernels: List[PortableKernel] = field(default_factory=list)
    accounting_warning: str = ""

    @property
    def total_instruction_ms(self) -> float:
        return self.delegated_ms + self.portable_ms

    @property
    def runtime_delegation_fraction(self) -> float:
        """Fraction of wall time spent inside XNNPACK, between 0.0 and 1.0."""
        if self.total_instruction_ms == 0.0:
            return 0.0
        return self.delegated_ms / self.total_instruction_ms

    def to_dict(self) -> dict:
        return {
            "method_execute_ms": self.method_execute_ms,
            "total_instruction_ms": self.total_instruction_ms,
            "delegated_ms": self.delegated_ms,
            "portable_ms": self.portable_ms,
            "runtime_delegation_fraction": self.runtime_delegation_fraction,
            "delegate_call_count": self.delegate_call_count,
            "operator_call_count": self.operator_call_count,
            "portable_kernels": [
                {
                    "name": kernel.name,
                    "operator_name": kernel.operator_name,
                    "total_ms": kernel.total_ms,
                    "call_count": kernel.call_count,
                    "runtime_fraction": kernel.runtime_fraction,
                }
                for kernel in self.portable_kernels
            ],
            "accounting_warning": self.accounting_warning,
        }


def collect_etdump(
    pte_path: str,
    input_path: str,
    etdump_runner_path: str,
    output_etdump_path: str,
    label: str,
    serial: str | None = None,
    iterations: int = 20,
    threads: int = 4,
) -> str:
    """Run the model on the device with tracing on, and pull back the ETDump.

    `label` distinguishes this run's files on the device. The before and after
    .pte files normally share a basename, so pushing them under their basenames
    would let one overwrite the other.
    """
    device.prepare_work_dir(serial=serial)
    device.push_runner(etdump_runner_path, serial=serial)

    remote_pte_name = f"{label}_profile_model.pte"
    remote_input_name = f"{label}_profile_input.bin"
    remote_dump_name = f"{label}_trace.etdump"
    device.push_file(pte_path, remote_pte_name, serial=serial)
    device.push_file(input_path, remote_input_name, serial=serial)

    command = (
        f"cd {device.DEVICE_WORK_DIR} && "
        f"./{device.ETDUMP_RUNNER_NAME} "
        f"--model_path {remote_pte_name} "
        f"--inputs {remote_input_name} "
        f"--num_executions {iterations} "
        f"--cpu_threads {threads} "
        f"--print_output none "
        f"--etdump_path {remote_dump_name}"
    )
    device.run_on_device(command, serial=serial)

    device.pull_file(
        f"{device.DEVICE_WORK_DIR}/{remote_dump_name}",
        output_etdump_path,
        serial=serial,
    )
    return output_etdump_path


def analyze_etdump(etdump_path: str) -> ProfileResult:
    """Turn an ETDump trace into a runtime breakdown."""
    from executorch.devtools import Inspector

    inspector = Inspector(etdump_path=etdump_path)
    table = inspector.to_dataframe()

    # Median, not mean, across the traced iterations. The runner has no warmup
    # flag, so iteration 1 includes lazy allocation and cold caches and can be
    # several times slower than steady state. Averaging lets that single
    # iteration swing the reported shares by tens of percent; the median
    # ignores it. Inspector computes both, so this costs nothing.
    TIME_COLUMN = "p50 (ms)"

    def total_ms_for(event_name: str) -> float:
        rows = table[table["event_name"] == event_name]
        return float(rows[TIME_COLUMN].sum())

    def count_for(event_name: str) -> int:
        return int((table["event_name"] == event_name).sum())

    delegated_ms = total_ms_for(DELEGATE_CALL_EVENT)
    portable_ms = total_ms_for(OPERATOR_CALL_EVENT)

    execute_rows = table[table["event_name"] == METHOD_EXECUTE_EVENT]
    if len(execute_rows) > 0:
        method_execute_ms = float(execute_rows[TIME_COLUMN].iloc[0])
    else:
        method_execute_ms = delegated_ms + portable_ms

    # Cross-check the nesting assumption described in the module docstring.
    accounting_warning = ""
    instruction_total = delegated_ms + portable_ms
    if method_execute_ms > 0:
        difference = abs(instruction_total - method_execute_ms) / method_execute_ms
        if difference > ACCOUNTING_TOLERANCE_FRACTION:
            accounting_warning = (
                f"Top-level instruction events total {instruction_total:.3f} ms but "
                f"Method::execute reports {method_execute_ms:.3f} ms "
                f"({100 * difference:.1f}% apart). Runtime-weighted delegation may "
                f"be inaccurate for this model."
            )

    # Per-kernel breakdown, from the nested native_call_* rows.
    portable_rows = table[table["event_name"].str.startswith(PORTABLE_KERNEL_PREFIX)]
    portable_kernels: List[PortableKernel] = []
    if len(portable_rows) > 0:
        grouped = portable_rows.groupby("event_name")[TIME_COLUMN].agg(["sum", "count"])
        for kernel_name, row in grouped.iterrows():
            total_kernel_ms = float(row["sum"])
            fraction = total_kernel_ms / instruction_total if instruction_total else 0.0
            portable_kernels.append(
                PortableKernel(
                    name=str(kernel_name),
                    total_ms=total_kernel_ms,
                    call_count=int(row["count"]),
                    runtime_fraction=fraction,
                )
            )

    # Most expensive first: this ordering is the hotspot ranking.
    portable_kernels.sort(key=lambda kernel: kernel.total_ms, reverse=True)

    return ProfileResult(
        method_execute_ms=method_execute_ms,
        delegated_ms=delegated_ms,
        portable_ms=portable_ms,
        delegate_call_count=count_for(DELEGATE_CALL_EVENT),
        operator_call_count=count_for(OPERATOR_CALL_EVENT),
        portable_kernels=portable_kernels,
        accounting_warning=accounting_warning,
    )


def profile_model(
    pte_path: str,
    input_path: str,
    etdump_runner_path: str,
    output_etdump_path: str,
    label: str,
    serial: str | None = None,
    iterations: int = 20,
    threads: int = 4,
) -> ProfileResult:
    """Collect and analyze an ETDump in one step.

    The absolute milliseconds here are higher than a clean benchmark, because
    the event tracer adds per-instruction overhead and because every traced
    iteration counts, including the cold first one. That is fine: profiling is
    used for attribution (which kernel owns which share of the time), and
    `benchmarking.py` supplies the latency numbers that decisions rest on.
    """
    collect_etdump(
        pte_path=pte_path,
        input_path=input_path,
        etdump_runner_path=etdump_runner_path,
        output_etdump_path=output_etdump_path,
        label=label,
        serial=serial,
        iterations=iterations,
        threads=threads,
    )
    return analyze_etdump(output_etdump_path)
