"""The accept/reject gate - DelegateDoctor's core philosophy in one function.

A repair is kept only when BOTH are true:

    1. it still computes the same thing  (numerical verification passed)
    2. it actually made inference faster (measured on the Arm target)

Neither condition alone is sufficient, and both failure modes were observed for
real during the feasibility study:

  * A rewrite that was *bit-exact* (max absolute error 0.0) and improved every
    structural metric - delegate blobs 4 -> 1, portable operators 3 -> 0,
    operator delegation 87.5% -> 100% - and ran 19% SLOWER, because XNNPACK's
    replacement kernel did real multiply-adds where the portable kernel had
    been doing a plain memory copy.

  * A rewrite that was dramatically faster and fully delegated, and silently
    corrupted 85% of the output pixels.

Note what is deliberately absent: delegation percentage plays no part in this
decision. It is diagnostic information, never an acceptance criterion.
"""

from __future__ import annotations

from dataclasses import dataclass

# Outcome strings, also used as the `decision` field in the JSON results.
ACCEPTED = "accepted"
REJECTED_VERIFICATION = "rejected_verification_failed"
REJECTED_PERFORMANCE = "rejected_no_performance_improvement"

# A recommendation, not a correctness threshold. Any measurable improvement is
# accepted by the gate; this constant only controls whether the report calls the
# win "modest". The feasibility study used 15-20% as a project-selection bar.
NOTEWORTHY_SPEEDUP = 1.15


def latency_reduction_percent(before_latency_ms: float, after_latency_ms: float) -> float:
    """How much of the original latency the repair removed, as a percentage.

    A 2.93x speedup removes 65.9% of the latency. Both numbers describe the same
    measurement, so the report states both rather than saying "193% faster",
    which readers interpret in at least two different ways.
    """
    if before_latency_ms <= 0:
        return 0.0
    return 100.0 * (before_latency_ms - after_latency_ms) / before_latency_ms


@dataclass
class RepairDecision:
    outcome: str
    speedup: float
    before_latency_ms: float
    after_latency_ms: float
    message: str

    @property
    def accepted(self) -> bool:
        return self.outcome == ACCEPTED

    @property
    def headline(self) -> str:
        """The single line printed at the end of a doctor run."""
        if self.outcome == ACCEPTED:
            return "REPAIR ACCEPTED"
        if self.outcome == REJECTED_VERIFICATION:
            return "REPAIR REJECTED - numerical verification failed"
        return "REPAIR REJECTED - no performance improvement"

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "accepted": self.accepted,
            "speedup": self.speedup,
            "before_latency_ms": self.before_latency_ms,
            "after_latency_ms": self.after_latency_ms,
            "message": self.message,
        }


def decide_repair(
    verification_passed: bool,
    before_latency_ms: float,
    after_latency_ms: float,
) -> RepairDecision:
    """Accept the repair only if it is both correct and faster.

    Latencies are p50 (median) on the Arm target, measured with the tracer-free
    runner so profiling instrumentation cannot influence the comparison.
    """
    if after_latency_ms <= 0.0:
        speedup = 0.0
    else:
        speedup = before_latency_ms / after_latency_ms

    # Correctness is checked first, and it is absolute. A faster wrong answer is
    # still a wrong answer.
    if not verification_passed:
        return RepairDecision(
            outcome=REJECTED_VERIFICATION,
            speedup=speedup,
            before_latency_ms=before_latency_ms,
            after_latency_ms=after_latency_ms,
            message=(
                "The repaired model does not reproduce the original outputs "
                "within tolerance. Performance is irrelevant when the answer "
                "is wrong, so the repair was discarded."
            ),
        )

    if after_latency_ms >= before_latency_ms:
        return RepairDecision(
            outcome=REJECTED_PERFORMANCE,
            speedup=speedup,
            before_latency_ms=before_latency_ms,
            after_latency_ms=after_latency_ms,
            message=(
                f"The repaired model is numerically correct but not faster "
                f"({before_latency_ms:.3f} ms -> {after_latency_ms:.3f} ms). "
                "Increased delegation on its own is not a reason to ship a "
                "change, so the repair was discarded."
            ),
        )

    # Report the speedup as a multiplier plus the latency reduction. "N% faster"
    # is ambiguous - readers disagree on whether it means the multiplier or the
    # share of time removed - so state both unambiguously.
    latency_reduction = latency_reduction_percent(before_latency_ms, after_latency_ms)

    if speedup < NOTEWORTHY_SPEEDUP:
        message = (
            f"The repaired model is numerically correct and achieves a "
            f"{speedup:.2f}x speedup ({latency_reduction:.1f}% lower p50 latency), "
            f"which is below the {NOTEWORTHY_SPEEDUP:.2f}x rule of thumb worth "
            f"acting on; confirm it is above measurement noise on your target "
            f"before relying on it."
        )
    else:
        message = (
            f"The repaired model is numerically correct and achieves a "
            f"{speedup:.2f}x speedup ({latency_reduction:.1f}% lower p50 latency)."
        )

    return RepairDecision(
        outcome=ACCEPTED,
        speedup=speedup,
        before_latency_ms=before_latency_ms,
        after_latency_ms=after_latency_ms,
        message=message,
    )
