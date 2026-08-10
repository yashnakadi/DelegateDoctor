"""DD-001 rewrite: does it preserve shape and value, and delegate-ably?

These tests need only torch.export, not an Arm device.
"""

import torch

from delegate_doctor.export_model import export_to_aten
from delegate_doctor.repairs import dd001_softmax


class SoftmaxOnChannels(torch.nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=1)


class ConvThenSoftmaxOnChannels(torch.nn.Module):
    """A conv in front matters.

    XNNPACK evaluates convolutions in NHWC. The naive 4-D permute form of this
    repair miscompiles in exactly this situation, so the rewrite is always
    exercised downstream of a conv.
    """

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 21, 1)

    def forward(self, x):
        return torch.softmax(self.conv(x), dim=1)


class SoftmaxRank7(torch.nn.Module):
    def forward(self, x):
        return torch.softmax(x.view(1, 1, 9, 8, 8, 4, 4), dim=2)


def _softmax_dims_in_graph(exported):
    """The (dim, input_rank) of every softmax left in a graph."""
    found = []
    for node in exported.graph.nodes:
        if node.op == "call_function" and node.target in dd001_softmax.SOFTMAX_TARGETS:
            rank = len(node.args[0].meta["val"].shape)
            found.append((int(node.args[1]), rank))
    return found


def test_rewrite_preserves_output_shape():
    example = (torch.randn(1, 21, 32, 32),)
    exported = export_to_aten(SoftmaxOnChannels().eval(), example)

    repaired_count = dd001_softmax.apply(exported)
    assert repaired_count == 1

    output = exported.module()(*example)
    assert tuple(output.shape) == (1, 21, 32, 32)


def test_rewritten_output_matches_the_original():
    torch.manual_seed(0)
    model = SoftmaxOnChannels().eval()
    example = (torch.randn(1, 21, 32, 32),)

    with torch.no_grad():
        expected = model(*example)

    exported = export_to_aten(model, example)
    dd001_softmax.apply(exported)
    actual = exported.module()(*example)

    # Same graph shape, same kernels, so this should be near bit-exact. The
    # end-to-end tolerance (1e-5) is checked by verification.py; here we assert
    # something much tighter because nothing should have changed numerically.
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0)


def test_rewrite_works_downstream_of_a_convolution():
    torch.manual_seed(0)
    model = ConvThenSoftmaxOnChannels().eval()
    example = (torch.randn(1, 8, 32, 32),)

    with torch.no_grad():
        expected = model(*example)

    exported = export_to_aten(model, example)
    assert dd001_softmax.apply(exported) == 1
    actual = exported.module()(*example)

    assert tuple(actual.shape) == tuple(expected.shape)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0)


def test_softmax_becomes_a_last_dimension_softmax():
    """The whole point of the repair: XNNPACK's constraint is now satisfied."""
    example = (torch.randn(1, 21, 32, 32),)
    exported = export_to_aten(SoftmaxOnChannels().eval(), example)

    before = _softmax_dims_in_graph(exported)
    assert before == [(1, 4)]  # dim 1 of a rank-4 tensor: XNNPACK refuses

    dd001_softmax.apply(exported)

    after = _softmax_dims_in_graph(exported)
    assert len(after) == 1
    dim, rank = after[0]
    # Collapsed to 3-D, softmax on the last axis.
    assert rank == 3
    assert dim == -1 or dim == rank - 1


def test_rewrite_uses_a_3d_reshape_not_a_4d_permute():
    """Regression guard for a silent miscompilation found during feasibility.

    Expressing the axis move as a 4-D permute is mathematically identical but
    produces wrong results on ExecuTorch 1.4.0 + XNNPACK when the input comes
    from a node evaluated in NHWC. Every permute this rewrite emits must
    therefore act on a rank-3 tensor.
    """
    example = (torch.randn(1, 8, 32, 32),)
    exported = export_to_aten(ConvThenSoftmaxOnChannels().eval(), example)
    dd001_softmax.apply(exported)

    permute_nodes = [
        node
        for node in exported.graph.nodes
        if node.op == "call_function"
        and node.target is torch.ops.aten.permute.default
    ]
    assert len(permute_nodes) == 2
    for node in permute_nodes:
        input_rank = len(node.args[0].meta["val"].shape)
        assert input_rank == 3, "DD-001 must never emit a rank-4 permute"


def test_rank7_rewrite_matches_the_original():
    torch.manual_seed(0)
    model = SoftmaxRank7().eval()
    example = (torch.randn(1, 9, 8, 8, 4, 4),)

    with torch.no_grad():
        expected = model(*example)

    exported = export_to_aten(model, example)
    assert dd001_softmax.apply(exported) == 1
    actual = exported.module()(*example)

    assert tuple(actual.shape) == tuple(expected.shape)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0)


def test_apply_is_a_no_op_when_nothing_is_detected():
    class AlreadyFine(torch.nn.Module):
        def forward(self, x):
            return torch.softmax(x, dim=-1)

    exported = export_to_aten(AlreadyFine().eval(), (torch.randn(1, 21, 32, 32),))
    assert dd001_softmax.apply(exported) == 0
