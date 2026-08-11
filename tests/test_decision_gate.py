"""The accept/reject gate - DelegateDoctor's core philosophy.

These tests are the most important in the suite. They encode the two failure
modes observed during the feasibility study, so that no future change can
quietly reintroduce either one.
"""

from delegate_doctor.decision import (
    ACCEPTED,
    REJECTED_DEVICE_VERIFICATION,
    REJECTED_PERFORMANCE,
    REJECTED_VERIFICATION,
    decide_repair,
    latency_reduction_percent,
)


# --- the four combinations -------------------------------------------------

def test_correct_and_faster_is_accepted():
    decision = decide_repair(
        host_verification_passed=True,
        device_verification_passed=True,
        before_latency_ms=76.5, after_latency_ms=26.4
    )
    assert decision.outcome == ACCEPTED
    assert decision.accepted
    assert decision.headline == "REPAIR ACCEPTED"
    assert decision.speedup > 2.8


def test_incorrect_but_faster_is_rejected():
    decision = decide_repair(
        host_verification_passed=False,
        device_verification_passed=True,
        before_latency_ms=76.5, after_latency_ms=1.6
    )
    assert decision.outcome == REJECTED_VERIFICATION
    assert not decision.accepted
    assert "numerical verification failed" in decision.headline


def test_correct_but_slower_is_rejected():
    decision = decide_repair(
        host_verification_passed=True,
        device_verification_passed=True,
        before_latency_ms=6.744, after_latency_ms=8.332
    )
    assert decision.outcome == REJECTED_PERFORMANCE
    assert not decision.accepted
    assert "no performance improvement" in decision.headline


def test_incorrect_and_slower_is_rejected():
    decision = decide_repair(
        host_verification_passed=False,
        device_verification_passed=True,
        before_latency_ms=10.0, after_latency_ms=20.0
    )
    assert not decision.accepted
    # Correctness is checked first, so that is the reason reported.
    assert decision.outcome == REJECTED_VERIFICATION


# --- the gate now has two correctness inputs, host and device --------------

def test_host_pass_device_pass_and_faster_is_accepted():
    decision = decide_repair(
        host_verification_passed=True,
        device_verification_passed=True,
        before_latency_ms=76.5,
        after_latency_ms=26.1,
    )
    assert decision.outcome == ACCEPTED


def test_host_fail_device_pass_and_faster_is_rejected():
    decision = decide_repair(
        host_verification_passed=False,
        device_verification_passed=True,
        before_latency_ms=76.5,
        after_latency_ms=26.1,
    )
    assert decision.outcome == REJECTED_VERIFICATION
    assert not decision.accepted


def test_host_pass_device_fail_and_faster_is_rejected():
    """The case host-only verification would have missed.

    A repair that looks correct on the development machine, fully delegates and
    runs nearly 3x faster is still discarded when the tensors the Android device
    actually produced do not verify.
    """
    decision = decide_repair(
        host_verification_passed=True,
        device_verification_passed=False,
        before_latency_ms=76.5,
        after_latency_ms=26.1,
    )
    assert decision.outcome == REJECTED_DEVICE_VERIFICATION
    assert not decision.accepted
    assert "Android numerical verification failed" in decision.headline
    assert decision.speedup > 2.9  # genuinely faster, and rejected anyway


def test_host_pass_device_pass_but_slower_is_rejected():
    decision = decide_repair(
        host_verification_passed=True,
        device_verification_passed=True,
        before_latency_ms=26.1,
        after_latency_ms=76.5,
    )
    assert decision.outcome == REJECTED_PERFORMANCE


def test_both_verifications_failing_reports_the_host_failure_first():
    decision = decide_repair(
        host_verification_passed=False,
        device_verification_passed=False,
        before_latency_ms=76.5,
        after_latency_ms=26.1,
    )
    assert decision.outcome == REJECTED_VERIFICATION


# --- regression tests for the feasibility findings -------------------------

def test_a_bit_exact_but_slower_repair_is_still_rejected():
    """Feasibility finding: structural improvement does not imply speed.

    The nearest-upsample repair was bit-exact (max absolute error 0.0) and took
    delegate blobs 4 -> 1, portable operators 3 -> 0, operator delegation
    87.5% -> 100%. It ran 19% slower and had to be discarded.
    """
    decision = decide_repair(
        host_verification_passed=True,
        device_verification_passed=True,
        before_latency_ms=6.744, after_latency_ms=8.332
    )
    assert not decision.accepted
    assert decision.speedup < 1.0


def test_a_dramatically_faster_wrong_repair_is_still_rejected():
    """Feasibility finding: a fast wrong answer is still a wrong answer.

    The naive 4-D permute form of DD-001 fully delegated and was ~30x faster
    while corrupting 85% of output pixels.
    """
    decision = decide_repair(
        host_verification_passed=False,
        device_verification_passed=True,
        before_latency_ms=48.3, after_latency_ms=1.6
    )
    assert not decision.accepted
    assert decision.speedup > 25  # it really was that much faster, and still rejected


def test_identical_latency_is_not_an_improvement():
    """Equal timings must not be accepted; ties go to the existing model."""
    decision = decide_repair(
        host_verification_passed=True,
        device_verification_passed=True,
        before_latency_ms=26.5, after_latency_ms=26.5
    )
    assert decision.outcome == REJECTED_PERFORMANCE


def test_delegation_is_not_an_input_to_the_decision():
    """Delegation is diagnostic only.

    decide_repair takes correctness and latency and nothing else. If a future
    change adds a delegation argument, this test should be revisited
    deliberately rather than by accident.
    """
    import inspect

    parameters = list(inspect.signature(decide_repair).parameters)
    assert parameters == [
        "host_verification_passed",
        "device_verification_passed",
        "before_latency_ms",
        "after_latency_ms",
    ]


# --- reporting behaviour ---------------------------------------------------

def test_small_improvement_is_accepted_but_flagged_as_modest():
    decision = decide_repair(
        host_verification_passed=True,
        device_verification_passed=True,
        before_latency_ms=100.0, after_latency_ms=99.0
    )
    assert decision.accepted
    assert "below" in decision.message


def test_speedup_is_reported_as_a_multiplier_and_a_latency_reduction():
    """"N% faster" is ambiguous, so the message must state both numbers."""
    decision = decide_repair(
        host_verification_passed=True,
        device_verification_passed=True,
        before_latency_ms=100.0, after_latency_ms=50.0
    )
    assert decision.accepted
    assert "2.00x speedup" in decision.message
    assert "50.0% lower p50 latency" in decision.message
    # The old ambiguous phrasing must not come back.
    assert "% faster" not in decision.message


def test_latency_reduction_percent_matches_the_speedup():
    # 2.93x speedup == 65.9% of the latency removed
    reduction = latency_reduction_percent(76.493, 26.114)
    assert round(reduction, 1) == 65.9


def test_latency_reduction_percent_handles_a_zero_baseline():
    assert latency_reduction_percent(0.0, 0.0) == 0.0
