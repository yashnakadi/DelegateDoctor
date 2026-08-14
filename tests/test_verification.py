"""The numerical gate must pass a good rewrite and fail a corrupted one."""

import torch

from delegate_doctor.verification import (
    MAX_ABSOLUTE_ERROR_TOLERANCE,
    compute_argmax_agreement,
    compute_error_metrics,
    verify_repair,
)


def make_probability_map(seed: int = 0) -> torch.Tensor:
    """A (1, 21, 16, 16) softmax output, like a small segmentation result."""
    torch.manual_seed(seed)
    logits = torch.randn(1, 21, 16, 16)
    return torch.softmax(logits, dim=1)


def test_identical_outputs_pass():
    output = make_probability_map()
    result = verify_repair(
        original_output=output,
        repaired_output=output.clone(),
        eager_output=output.clone(),
        argmax_dim=1,
    )
    assert result.passed
    assert result.status_text == "PASS"
    assert result.argmax_agreement == 1.0
    assert result.failure_reasons == []


def test_kernel_rounding_noise_passes():
    """Differences of the size fp32 kernels actually produce must not fail."""
    original = make_probability_map()
    repaired = original + torch.full_like(original, 2e-8)

    result = verify_repair(
        original_output=original,
        repaired_output=repaired,
        eager_output=original,
        argmax_dim=1,
    )
    assert result.passed
    assert result.repaired_vs_original.max_absolute_error < MAX_ABSOLUTE_ERROR_TOLERANCE


def test_error_just_above_tolerance_fails():
    original = make_probability_map()
    corrupted = original.clone()
    corrupted[0, 0, 0, 0] += MAX_ABSOLUTE_ERROR_TOLERANCE * 10

    result = verify_repair(
        original_output=original,
        repaired_output=corrupted,
        eager_output=original,
        argmax_dim=None,
    )
    assert not result.passed
    assert result.status_text == "FAIL"
    assert any("tolerance" in reason for reason in result.failure_reasons)


def test_transposed_output_fails():
    """The exact corruption the naive 4-D permute repair produced.

    Swapping two spatial axes leaves the value distribution untouched, so
    summary statistics look plausible; only an elementwise comparison catches
    it. This is the regression test for that feasibility finding.
    """
    original = make_probability_map()
    corrupted = original.transpose(2, 3).contiguous()

    result = verify_repair(
        original_output=original,
        repaired_output=corrupted,
        eager_output=original,
        argmax_dim=1,
    )
    assert not result.passed
    assert result.argmax_agreement < 1.0


def test_argmax_disagreement_alone_fails_even_when_error_is_tiny():
    """A change too small to breach the error budget can still flip a class.

    Two nearly-tied classes are swapped by a nudge of 1e-6. Absolute error stays
    inside tolerance, but the predicted class changes, which is what a
    segmentation consumer would actually see.
    """
    original = torch.zeros(1, 2, 1, 1)
    original[0, 0, 0, 0] = 0.5000000
    original[0, 1, 0, 0] = 0.4999995

    repaired = original.clone()
    repaired[0, 0, 0, 0] = 0.4999995
    repaired[0, 1, 0, 0] = 0.5000000

    result = verify_repair(
        original_output=original,
        repaired_output=repaired,
        eager_output=original,
        argmax_dim=1,
    )
    assert result.repaired_vs_original.max_absolute_error < MAX_ABSOLUTE_ERROR_TOLERANCE
    assert result.argmax_agreement == 0.0
    assert not result.passed


def test_shape_mismatch_raises_a_clear_error():
    original = make_probability_map()
    wrong_shape = torch.zeros(1, 21, 8, 8)
    try:
        compute_error_metrics(wrong_shape, original)
    except ValueError as error:
        assert "different shapes" in str(error)
    else:
        raise AssertionError("expected a ValueError for mismatched shapes")


def test_argmax_agreement_is_a_fraction():
    a = torch.tensor([[[[1.0]], [[0.0]]]])   # argmax along dim 1 -> 0
    b = torch.tensor([[[[0.0]], [[1.0]]]])   # argmax along dim 1 -> 1
    assert compute_argmax_agreement(a, a, dim=1) == 1.0
    assert compute_argmax_agreement(a, b, dim=1) == 0.0


# --- backend fidelity is not a repair failure ---------------------------------

def test_a_backend_that_already_drifts_from_eager_is_not_the_repairs_fault():
    """The measured Inception V3 case.

    ExecuTorch's own output sits 1.43e-05 from PyTorch eager *before* any
    repair, and the candidate sits exactly the same distance away. The repair's
    own comparison is 2.62e-06, well inside tolerance. Folding the eager
    comparison into `passed` rejected that repair and called it a host
    correctness failure.
    """
    torch.manual_seed(0)
    eager = torch.randn(1, 1000)
    # Both ExecuTorch programs drift from eager by the same amount, and agree
    # with each other closely.
    original = eager + 1.43e-05
    repaired = original + 2.62e-06

    result = verify_repair(original_output=original, repaired_output=repaired,
                           eager_output=eager, argmax_dim=1)

    assert result.passed, "the repair reproduces the original ExecuTorch output"
    assert result.failure_reasons == []
    assert result.backend_fidelity == "WARNING"
    assert result.backend_fidelity_acceptable
    assert result.original_vs_eager is not None


def test_a_candidate_that_walks_away_from_eager_alone_fails():
    """The original agrees with eager; the candidate does not."""
    torch.manual_seed(0)
    eager = torch.randn(1, 1000)
    original = eager.clone()
    repaired = eager + 0.4

    result = verify_repair(original_output=original, repaired_output=repaired,
                           eager_output=eager, argmax_dim=1)

    # Repair semantics catch it first, and backend fidelity blames it correctly.
    assert not result.passed
    assert result.backend_fidelity == "FAIL"
    assert not result.backend_fidelity_acceptable


def test_repair_semantics_still_fail_on_a_genuinely_wrong_rewrite():
    """Nothing about backend fidelity softens the repair's own gate."""
    torch.manual_seed(0)
    original = torch.randn(1, 1000)
    repaired = original + 0.5

    result = verify_repair(original_output=original, repaired_output=repaired,
                           eager_output=None, argmax_dim=1)

    assert not result.passed
    assert any("differs from the original" in reason
               for reason in result.failure_reasons)


def test_the_host_tolerance_is_unchanged():
    from delegate_doctor import verification

    assert verification.MAX_ABSOLUTE_ERROR_TOLERANCE == 1e-5
    assert verification.REQUIRED_ARGMAX_AGREEMENT == 1.0
