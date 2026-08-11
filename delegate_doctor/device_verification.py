"""Verify the repair using tensors produced by the Android device itself.

Why this exists
---------------
Host verification runs both .pte files through ExecuTorch's Python runtime on
the development machine. That catches a wrong rewrite, but it cannot catch a
wrong *backend*: the Android build of XNNPACK is different compiled code on a
different architecture.

This is not a hypothetical gap. During earlier work a rewrite that was
mathematically identical, fully delegated, and dramatically faster turned out to
miscompile inside XNNPACK, corrupting 85% of output pixels. A graph rewrite can
be correct on the host and still trigger a device-specific bug, so DelegateDoctor
retrieves the real Android output and checks it before accepting a repair.

How the output is captured
--------------------------
The stock ExecuTorch `executor_runner` already supports `--output_file`, which
writes each output tensor as raw contiguous bytes to `<name>-<index>.bin`. No
custom runner and no patch to ExecuTorch are needed: we reuse the tracer-free
benchmark runner in a *separate, untimed* invocation.

    push original.pte + the same input bytes the host used
        -> run once with --output_file before_output
        -> adb pull before_output-0.bin
    same for the repaired model

That raw file carries no dtype or shape, so it is never interpreted on its own.
The host already knows what the output must look like (it just ran the same
model), so the expected dtype, shape and byte count come from the host tensor
and the device file is validated against them. A size that does not match is an
error, never a reshape.

Benchmark integrity
-------------------
This runs before or after the timed benchmark, never inside it. The benchmark
invocation still uses `--print_output none` and writes no tensors, so file I/O
and adb transfer cannot leak into latency numbers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

from . import device
from .verification import (
    MAX_ABSOLUTE_ERROR_TOLERANCE,
    REQUIRED_ARGMAX_AGREEMENT,
    ErrorMetrics,
    compute_argmax_agreement,
    compute_error_metrics,
)

# Only fp32 is supported, matching the project's declared scope. The device file
# is raw bytes, so guessing at any other dtype would silently produce garbage.
SUPPORTED_DTYPE = torch.float32
BYTES_PER_ELEMENT = 4

# `--output_file NAME` makes the runner write NAME-<output index>.bin.
# DelegateDoctor verifies the first output tensor only.
FIRST_OUTPUT_INDEX = 0


class DeviceVerificationError(RuntimeError):
    """Raised when a device output cannot be retrieved or trusted."""


@dataclass
class TensorSpec:
    """What the device output is required to be, taken from the host tensor."""

    dtype: torch.dtype
    shape: tuple
    element_count: int

    @property
    def expected_bytes(self) -> int:
        return self.element_count * BYTES_PER_ELEMENT

    def describe(self) -> str:
        shape_text = ",".join(str(size) for size in self.shape)
        return (
            f"dtype=float32\n"
            f"shape={shape_text}\n"
            f"count={self.element_count}\n"
            f"bytes={self.expected_bytes}\n"
        )


def spec_from_host_tensor(host_tensor: torch.Tensor) -> TensorSpec:
    """Derive the expected device output layout from the host result."""
    if host_tensor.dtype != SUPPORTED_DTYPE:
        raise DeviceVerificationError(
            f"Device verification supports float32 outputs only, but this model's "
            f"first output is {host_tensor.dtype}.\n"
            f"The device writes raw bytes with no dtype tag, so reinterpreting "
            f"them would silently produce wrong numbers."
        )
    return TensorSpec(
        dtype=SUPPORTED_DTYPE,
        shape=tuple(int(size) for size in host_tensor.shape),
        element_count=int(host_tensor.numel()),
    )


def load_device_tensor(path: str, spec: TensorSpec) -> torch.Tensor:
    """Read a raw device output file and check it against the expected spec.

    Every failure here is a verification failure, never a best-effort reshape.
    """
    if not os.path.isfile(path):
        raise DeviceVerificationError(
            f"The device did not produce an output tensor.\n"
            f"Expected file: {path}"
        )

    actual_bytes = os.path.getsize(path)
    if actual_bytes == 0:
        raise DeviceVerificationError(
            f"The device output file is empty: {path}\n"
            f"Expected {spec.expected_bytes} bytes "
            f"({spec.element_count} float32 values)."
        )
    if actual_bytes != spec.expected_bytes:
        raise DeviceVerificationError(
            f"The device output file has the wrong size.\n"
            f"  file:     {path}\n"
            f"  expected: {spec.expected_bytes} bytes "
            f"({spec.element_count} float32 values, shape {spec.shape})\n"
            f"  actual:   {actual_bytes} bytes\n"
            f"The output was not reshaped; this is treated as a failure."
        )

    import numpy

    values = numpy.fromfile(path, dtype=numpy.float32)
    if values.size != spec.element_count:
        raise DeviceVerificationError(
            f"The device output has {values.size} float32 values but "
            f"{spec.element_count} were expected ({path})."
        )
    return torch.from_numpy(values).reshape(spec.shape)


def write_spec_sidecar(spec: TensorSpec, path: str) -> str:
    """Save the tensor metadata next to the pulled bytes, so the run artifacts
    are self-describing rather than a bare unlabelled blob."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as sidecar:
        sidecar.write(spec.describe())
    return path


def capture_device_output(
    pte_path: str,
    input_paths,
    bench_runner_path: str,
    label: str,
    output_dir: str,
    serial: str,
    threads: int = 4,
) -> str:
    """Run a model once on the device and pull back its first output tensor.

    `label` ("before" / "after") keeps every remote and local filename distinct,
    so the original and repaired outputs cannot overwrite one another.
    """
    device.prepare_work_dir(serial=serial)
    device.push_runner(bench_runner_path, serial=serial)

    if isinstance(input_paths, str):
        input_paths = [input_paths]
    remote_pte = f"{label}_verify_model.pte"
    remote_output_base = f"{label}_output"
    remote_output_file = f"{remote_output_base}-{FIRST_OUTPUT_INDEX}.bin"

    device.push_file(pte_path, remote_pte, serial=serial)
    # executor_runner takes a comma-separated list, one file per positional input.
    remote_inputs = []
    for index, local_input in enumerate(input_paths):
        remote_name = f"{label}_verify_input{index}.bin"
        device.push_file(local_input, remote_name, serial=serial)
        remote_inputs.append(remote_name)
    remote_input = ",".join(remote_inputs)
    # Remove any stale file so a failed run cannot be mistaken for a fresh one.
    device.remove_remote_files(f"{remote_output_base}-*.bin", serial=serial)

    command = (
        f"cd {device.DEVICE_WORK_DIR} && "
        f"./{device.BENCH_RUNNER_NAME} "
        f"--model_path {remote_pte} "
        f"--inputs {remote_input} "
        f"--num_executions 1 "
        f"--cpu_threads {threads} "
        f"--print_output none "
        f"--output_file {remote_output_base}"
    )
    try:
        device.run_on_device(command, serial=serial)
    except device.DeviceError as error:
        raise DeviceVerificationError(
            f"Android verification failed: the {label} model did not run on the "
            f"device.\n{error}"
        )

    local_path = os.path.join(output_dir, f"{label}_device_output.bin")
    remote_path = f"{device.DEVICE_WORK_DIR}/{remote_output_file}"
    try:
        device.pull_file(remote_path, local_path, serial=serial)
    except Exception:
        raise DeviceVerificationError(
            f"Android verification failed: the {label} model produced no output "
            f"tensor.\n"
            f"  expected on device: {remote_path}\n"
            f"  expected locally:   {local_path}"
        )

    device.remove_remote_files(f"{remote_output_base}-*.bin", serial=serial)
    return local_path


@dataclass
class DeviceVerificationResult:
    """Outcome of comparing the tensors the Android device actually produced."""

    passed: bool
    # repaired vs original, both measured on the device: the main question
    repaired_vs_original: Optional[ErrorMetrics] = None
    argmax_agreement: Optional[float] = None
    # device vs host for each model: distinguishes "bad repair" from "bad backend"
    original_device_vs_host: Optional[ErrorMetrics] = None
    repaired_device_vs_host: Optional[ErrorMetrics] = None
    failure_reasons: list = None
    error: str = ""

    def __post_init__(self):
        if self.failure_reasons is None:
            self.failure_reasons = []

    @property
    def status_text(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def to_dict(self) -> dict:
        def metrics_or_none(metrics):
            return metrics.to_dict() if metrics is not None else None

        return {
            "passed": self.passed,
            "repaired_vs_original": metrics_or_none(self.repaired_vs_original),
            "argmax_agreement": self.argmax_agreement,
            "original_device_vs_host": metrics_or_none(self.original_device_vs_host),
            "repaired_device_vs_host": metrics_or_none(self.repaired_device_vs_host),
            "failure_reasons": self.failure_reasons,
            "error": self.error,
        }


def verify_device_outputs(
    original_device_output: torch.Tensor,
    repaired_device_output: torch.Tensor,
    original_host_output: torch.Tensor,
    repaired_host_output: torch.Tensor,
    argmax_dim: Optional[int] = None,
) -> DeviceVerificationResult:
    """Compare the device tensors, using the same thresholds as host verification.

    Three checks, all of which must pass:

      1. repaired vs original, both on the device - the repair itself;
      2. original device vs original host - does the Android backend already
         disagree with the host before we changed anything?
      3. repaired device vs repaired host - did the rewrite trigger a
         device-specific bug?

    Checks 2 and 3 are what separate "the repair is wrong" from "the backend is
    wrong", and either failing is treated as a correctness failure. That is the
    conservative choice: a device that does not reproduce its own host result is
    not a device whose speedup we should trust.
    """
    # Note the thresholds and metric helpers come straight from verification.py:
    # host and device share one correctness policy, never two.
    failure_reasons = []

    for name, tensor in (
        ("original", original_device_output),
        ("repaired", repaired_device_output),
    ):
        expected_shape = (
            original_host_output.shape if name == "original" else repaired_host_output.shape
        )
        if tuple(tensor.shape) != tuple(expected_shape):
            failure_reasons.append(
                f"the {name} device output has shape {tuple(tensor.shape)}, "
                f"expected {tuple(expected_shape)}"
            )

    if failure_reasons:
        return DeviceVerificationResult(passed=False, failure_reasons=failure_reasons)

    repaired_vs_original = compute_error_metrics(
        repaired_device_output, original_device_output
    )
    if repaired_vs_original.max_absolute_error > MAX_ABSOLUTE_ERROR_TOLERANCE:
        failure_reasons.append(
            f"on the device, the repaired output differs from the original by "
            f"{repaired_vs_original.max_absolute_error:.3e}, above the tolerance "
            f"of {MAX_ABSOLUTE_ERROR_TOLERANCE:g}"
        )

    original_device_vs_host = compute_error_metrics(
        original_device_output, original_host_output
    )
    if original_device_vs_host.max_absolute_error > MAX_ABSOLUTE_ERROR_TOLERANCE:
        failure_reasons.append(
            f"the original model's device output differs from its host output by "
            f"{original_device_vs_host.max_absolute_error:.3e}; the Android "
            f"backend does not reproduce the host result even before the repair"
        )

    repaired_device_vs_host = compute_error_metrics(
        repaired_device_output, repaired_host_output
    )
    if repaired_device_vs_host.max_absolute_error > MAX_ABSOLUTE_ERROR_TOLERANCE:
        failure_reasons.append(
            f"the repaired model's device output differs from its host output by "
            f"{repaired_device_vs_host.max_absolute_error:.3e}; the rewrite looks "
            f"correct on the host but not on the Android backend"
        )

    argmax_agreement = None
    if argmax_dim is not None:
        argmax_agreement = compute_argmax_agreement(
            repaired_device_output, original_device_output, argmax_dim
        )
        if argmax_agreement < REQUIRED_ARGMAX_AGREEMENT:
            failure_reasons.append(
                f"on the device, argmax agreement is "
                f"{100 * argmax_agreement:.4f}%, below the required "
                f"{100 * REQUIRED_ARGMAX_AGREEMENT:.4f}%"
            )

    return DeviceVerificationResult(
        passed=len(failure_reasons) == 0,
        repaired_vs_original=repaired_vs_original,
        argmax_agreement=argmax_agreement,
        original_device_vs_host=original_device_vs_host,
        repaired_device_vs_host=repaired_device_vs_host,
        failure_reasons=failure_reasons,
    )


def run_device_verification(
    before_pte_path: str,
    after_pte_path: str,
    input_paths,
    bench_runner_path: str,
    original_host_output: torch.Tensor,
    repaired_host_output: torch.Tensor,
    output_dir: str,
    serial: str,
    argmax_dim: Optional[int] = None,
    threads: int = 4,
) -> DeviceVerificationResult:
    """Capture both device outputs and verify them. The whole step, end to end."""
    spec = spec_from_host_tensor(original_host_output)
    write_spec_sidecar(spec, os.path.join(output_dir, "device_output.meta.txt"))

    original_path = capture_device_output(
        pte_path=before_pte_path,
        input_paths=input_paths,
        bench_runner_path=bench_runner_path,
        label="before",
        output_dir=output_dir,
        serial=serial,
        threads=threads,
    )
    repaired_path = capture_device_output(
        pte_path=after_pte_path,
        input_paths=input_paths,
        bench_runner_path=bench_runner_path,
        label="after",
        output_dir=output_dir,
        serial=serial,
        threads=threads,
    )

    original_device_output = load_device_tensor(original_path, spec)
    repaired_device_output = load_device_tensor(
        repaired_path, spec_from_host_tensor(repaired_host_output)
    )

    return verify_device_outputs(
        original_device_output=original_device_output,
        repaired_device_output=repaired_device_output,
        original_host_output=original_host_output,
        repaired_host_output=repaired_host_output,
        argmax_dim=argmax_dim,
    )
