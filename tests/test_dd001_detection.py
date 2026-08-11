"""DD-001 detection: does it find the right softmax nodes, and only those?

These tests need only torch.export, not an Arm device.
"""

import torch

from delegate_doctor.export_model import export_to_aten
from delegate_doctor.repairs import dd001_softmax


class SoftmaxOnChannels(torch.nn.Module):
    """Rank-4 (N, C, H, W) softmax over the class dimension - the U-Net case."""

    def forward(self, x):
        return torch.softmax(x, dim=1)


class SoftmaxOnLastDim(torch.nn.Module):
    """Rank-4 softmax on the last dimension - XNNPACK already accepts this."""

    def forward(self, x):
        return torch.softmax(x, dim=-1)


class SoftmaxRank7(torch.nn.Module):
    """Rank-7 softmax over dim 2.

    This is the shape torchvision's RAFT optical-flow model produces in its
    convex-upsampling step, so it is a real pattern and not just a stress test.
    """

    def forward(self, x):
        reshaped = x.view(1, 1, 9, 8, 8, 4, 4)
        return torch.softmax(reshaped, dim=2)


class SoftmaxRank1(torch.nn.Module):
    """Rank-1 softmax: too few dimensions for the rewrite to be meaningful."""

    def forward(self, x):
        return torch.softmax(x, dim=0)


def test_rank4_channel_softmax_is_detected():
    exported = export_to_aten(SoftmaxOnChannels().eval(), (torch.randn(1, 21, 32, 32),))
    result = dd001_softmax.detect(exported)

    assert result.applies
    assert len(result.detections) == 1

    detection = result.detections[0]
    assert detection.tensor_rank == 4
    assert detection.softmax_dim == 1
    assert detection.last_dim == 3
    assert detection.input_shape == (1, 21, 32, 32)
    # 32*32 = 1024 vectors of 21 elements, each 1024 elements apart.
    assert detection.vector_length == 21
    assert detection.vector_count == 1024
    assert detection.element_stride == 1024


def test_rank7_non_last_softmax_is_detected():
    exported = export_to_aten(SoftmaxRank7().eval(), (torch.randn(1, 9, 8, 8, 4, 4),))
    result = dd001_softmax.detect(exported)

    assert result.applies
    detection = result.detections[0]
    assert detection.tensor_rank == 7
    assert detection.softmax_dim == 2
    assert detection.last_dim == 6


def test_last_dimension_softmax_is_not_detected():
    exported = export_to_aten(SoftmaxOnLastDim().eval(), (torch.randn(1, 21, 32, 32),))
    result = dd001_softmax.detect(exported)

    assert not result.applies
    assert len(result.detections) == 0
    # It should be skipped with an explanation, not silently ignored.
    assert len(result.skipped) == 1
    assert "already on the last dimension" in result.skipped[0].reason


def test_unsupported_rank_is_rejected_with_an_explanation():
    exported = export_to_aten(SoftmaxRank1().eval(), (torch.randn(16),))
    result = dd001_softmax.detect(exported)

    assert not result.applies
    assert len(result.skipped) == 1
    assert "rank" in result.skipped[0].reason


def test_dynamic_shape_is_rejected_rather_than_guessed():
    """A symbolic dimension means the flattened sizes cannot be computed.

    DD-001 must decline rather than bake in a size it only saw once.
    """
    model = SoftmaxOnChannels().eval()
    # Batch size 2, not 1: torch.export specializes a size-1 dimension to a
    # constant, which would defeat the point of the test.
    example = (torch.randn(2, 21, 32, 32),)
    batch = torch.export.Dim("batch", min=2, max=8)
    exported = torch.export.export(
        model, example, dynamic_shapes={"x": {0: batch}}, strict=True
    )

    result = dd001_softmax.detect(exported)

    assert not result.applies
    assert len(result.skipped) == 1
    assert "dynamic" in result.skipped[0].reason


def test_detection_explanation_mentions_the_key_facts():
    exported = export_to_aten(SoftmaxOnChannels().eval(), (torch.randn(1, 21, 32, 32),))
    explanation = dd001_softmax.detect(exported).detections[0].explain()

    assert "softmax(dim=1)" in explanation
    assert "[1, 21, 32, 32]" in explanation
    assert "rank 4" in explanation
    assert "last dim 3" in explanation
