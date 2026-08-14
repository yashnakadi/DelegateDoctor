"""The numerical gate: did the repair change what the model computes?

This gate is mandatory and it is not a formality. During the feasibility study a
repair variant was produced that:

  * increased delegation,
  * removed every portable operator,
  * benchmarked dramatically faster,

and silently corrupted the output - only 15.3% of pixels kept their correct
predicted class. Every structural signal said "success". Only comparing the
actual tensors caught it.

So: a repair is never accepted on the strength of delegation or speed.

Tolerances
----------
The DD-001 rewrite is exact in real arithmetic, but not bit-exact in fp32,
because the original and repaired graphs use different kernels (ExecuTorch's
portable reference softmax versus XNNPACK's vectorised one) with different
`exp` implementations and different reduction orders.

The thresholds live at the top of this file, on purpose, so they are easy to
find and change.

Two questions, not one
----------------------
The same distinction the device path draws, drawn here:

    REPAIR SEMANTICS    original ExecuTorch output vs candidate ExecuTorch
                        output. Did the rewrite change what the model
                        computes? This is `passed`.

    BACKEND FIDELITY    each program's ExecuTorch output vs PyTorch eager. How
                        closely does this backend reproduce PyTorch? Measured
                        for the original as well as the candidate, and
                        classified by `classify_backend_fidelity`.

The eager comparison used to be folded into `passed`, which meant any model
whose ExecuTorch build already drifted from PyTorch had every repair rejected
and reported as a host correctness failure. That is not a fact about the
rewrite, and the rewrite is not what it should have accused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Thresholds. Deliberately at module top level so they are easy to locate.
# ---------------------------------------------------------------------------

# Maximum absolute difference allowed between the original and repaired
# outputs. 1e-5 is roughly 100x fp32 epsilon: comfortably above kernel rounding
# noise, comfortably below any difference that could change a decision.
MAX_ABSOLUTE_ERROR_TOLERANCE = 1e-5

# When the model has a meaningful class dimension, we additionally require that
# every element still picks the same class. For a segmentation model this means
# every pixel keeps its predicted label.
REQUIRED_ARGMAX_AGREEMENT = 1.0


@dataclass
class ErrorMetrics:
    """Difference between two tensors."""

    max_absolute_error: float
    mean_absolute_error: float
    mean_squared_error: float
    max_relative_error: float

    def to_dict(self) -> dict:
        return {
            "max_absolute_error": self.max_absolute_error,
            "mean_absolute_error": self.mean_absolute_error,
            "mean_squared_error": self.mean_squared_error,
            "max_relative_error": self.max_relative_error,
        }


@dataclass
class VerificationResult:
    # Repair semantics only: does the candidate still compute what the original
    # ExecuTorch program computed? Agreement with PyTorch eager is a separate
    # question - see `backend_fidelity`.
    passed: bool
    repaired_vs_original: ErrorMetrics
    repaired_vs_eager: Optional[ErrorMetrics] = None
    original_vs_eager: Optional[ErrorMetrics] = None
    backend_fidelity: str = "OK"
    backend_fidelity_reason: str = ""
    argmax_agreement: Optional[float] = None
    failure_reasons: list = field(default_factory=list)

    @property
    def status_text(self) -> str:
        return "PASS" if self.passed else "FAIL"

    @property
    def backend_fidelity_acceptable(self) -> bool:
        return self.backend_fidelity != "FAIL"

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "max_absolute_error_tolerance": MAX_ABSOLUTE_ERROR_TOLERANCE,
            "required_argmax_agreement": REQUIRED_ARGMAX_AGREEMENT,
            "repaired_vs_original": self.repaired_vs_original.to_dict(),
            "repaired_vs_eager": (
                self.repaired_vs_eager.to_dict() if self.repaired_vs_eager else None
            ),
            "original_vs_eager": (
                self.original_vs_eager.to_dict() if self.original_vs_eager else None
            ),
            "backend_fidelity": self.backend_fidelity,
            "backend_fidelity_reason": self.backend_fidelity_reason,
            "backend_fidelity_acceptable": self.backend_fidelity_acceptable,
            "argmax_agreement": self.argmax_agreement,
            "failure_reasons": self.failure_reasons,
        }


def compute_error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> ErrorMetrics:
    """Compare two tensors of the same shape."""
    if actual.shape != expected.shape:
        raise ValueError(
            f"Cannot compare tensors of different shapes: "
            f"{tuple(actual.shape)} vs {tuple(expected.shape)}"
        )

    difference = (actual - expected).abs()

    # Guard the denominator so a near-zero expected value cannot turn a tiny
    # absolute difference into a meaningless huge relative error.
    safe_denominator = expected.abs().clamp_min(1e-6)

    return ErrorMetrics(
        max_absolute_error=difference.max().item(),
        mean_absolute_error=difference.mean().item(),
        mean_squared_error=((actual - expected) ** 2).mean().item(),
        max_relative_error=(difference / safe_denominator).max().item(),
    )


def compute_argmax_agreement(
    actual: torch.Tensor, expected: torch.Tensor, dim: int
) -> float:
    """Fraction of positions where both tensors pick the same index along `dim`."""
    matches = actual.argmax(dim) == expected.argmax(dim)
    return matches.float().mean().item()


def verify_repair(
    original_output: torch.Tensor,
    repaired_output: torch.Tensor,
    eager_output: Optional[torch.Tensor] = None,
    argmax_dim: Optional[int] = None,
) -> VerificationResult:
    """Decide whether the repaired model still computes the original function.

    `original_output` and `repaired_output` are the outputs of the two
    ExecuTorch programs; comparing them is the repair's own gate, and it is
    what `passed` reports.

    `eager_output` is the plain PyTorch result. Comparing an ExecuTorch program
    against eager measures the *backend*, not the rewrite: ExecuTorch and
    XNNPACK use different kernels and different reduction orders from PyTorch,
    and on a deep network that difference accumulates. Inception V3's untouched
    original diverges 1.43e-05 from eager before DelegateDoctor changes
    anything, and its candidate diverges by exactly the same amount - so
    treating that as a repair failure rejected a repair whose own comparison
    was 2.62e-06, well inside tolerance.

    So the eager comparison is made for both programs and classified as backend
    fidelity, with the same policy the device path uses: a discrepancy the
    candidate introduced, or one dramatically worse than the original's, is
    still a failure. A pre-existing one is a warning about the backend.
    """
    from .device_verification import classify_backend_fidelity

    failure_reasons = []

    repaired_vs_original = compute_error_metrics(repaired_output, original_output)
    if repaired_vs_original.max_absolute_error > MAX_ABSOLUTE_ERROR_TOLERANCE:
        failure_reasons.append(
            f"repaired output differs from the original ExecuTorch output by "
            f"{repaired_vs_original.max_absolute_error:.3e}, "
            f"above the tolerance of {MAX_ABSOLUTE_ERROR_TOLERANCE:g}"
        )

    repaired_vs_eager = None
    original_vs_eager = None
    backend_fidelity, backend_fidelity_reason = "OK", ""
    if eager_output is not None:
        repaired_vs_eager = compute_error_metrics(repaired_output, eager_output)
        original_vs_eager = compute_error_metrics(original_output, eager_output)
        backend_fidelity, backend_fidelity_reason = classify_backend_fidelity(
            original_vs_eager.max_absolute_error,
            repaired_vs_eager.max_absolute_error,
        )

    argmax_agreement = None
    if argmax_dim is not None:
        argmax_agreement = compute_argmax_agreement(
            repaired_output, original_output, argmax_dim
        )
        if argmax_agreement < REQUIRED_ARGMAX_AGREEMENT:
            failure_reasons.append(
                f"argmax agreement is {100 * argmax_agreement:.4f}%, "
                f"below the required {100 * REQUIRED_ARGMAX_AGREEMENT:.4f}%"
            )

    return VerificationResult(
        passed=len(failure_reasons) == 0,
        repaired_vs_original=repaired_vs_original,
        repaired_vs_eager=repaired_vs_eager,
        original_vs_eager=original_vs_eager,
        backend_fidelity=backend_fidelity,
        backend_fidelity_reason=backend_fidelity_reason,
        argmax_agreement=argmax_agreement,
        failure_reasons=failure_reasons,
    )
