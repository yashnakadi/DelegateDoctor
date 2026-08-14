"""Tests for DD-003 - avg_pool2d padding made explicit.

Offline: torch.export only. No adb, device, NDK, network or native build.

The centre of gravity here is equivalence. DD-003 claims the rewrite is
value-identical, not merely close, so most of these tests compare tensors
against exactly 0.0 rather than against a tolerance.
"""

import copy

import pytest
import torch

from delegate_doctor.export_model import export_to_aten
from delegate_doctor.repairs import ALL_RULES, dd002_noop_alias, dd003_avgpool_pad


class Pool(torch.nn.Module):
    """One avg_pool2d, every argument controllable."""

    def __init__(self, kernel_size=3, stride=1, padding=1, ceil_mode=False,
                 count_include_pad=True, divisor_override=None):
        super().__init__()
        self.settings = dict(
            kernel_size=kernel_size, stride=stride, padding=padding,
            ceil_mode=ceil_mode, count_include_pad=count_include_pad,
            divisor_override=divisor_override)

    def forward(self, x):
        return torch.nn.functional.avg_pool2d(x, **self.settings)


class DefaultedPool(torch.nn.Module):
    """The Inception form: the arguments that matter are never mentioned.

    `ceil_mode` and `count_include_pad` are left at their defaults, so
    torch.export omits them from `node.args` entirely.
    """

    def forward(self, x):
        return torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1,
                                              padding=1)


class NoPooling(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)


class TwoSites(torch.nn.Module):
    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 3, 1, 1)
        return torch.nn.functional.avg_pool2d(x, 3, 1, 1)


def export(module, shape=(1, 8, 16, 16)):
    return export_to_aten(module.eval(), (torch.randn(*shape),))


def count_pools(exported):
    return sum(
        1
        for node in exported.graph.nodes
        if node.op == "call_function" and node.target in dd003_avgpool_pad.AVGPOOL_TARGETS
    )


def count_pads(exported):
    return sum(
        1
        for node in exported.graph.nodes
        if node.op == "call_function"
        and node.target is torch.ops.aten.constant_pad_nd.default
    )


# --- detection ----------------------------------------------------------------


def test_detects_the_inception_configuration():
    result = dd003_avgpool_pad.detect(export(Pool()))
    assert result.applies
    assert len(result.detections) == 1
    assert result.detections[0].padding == (1, 1)


def test_detects_when_the_deciding_arguments_are_left_at_their_defaults():
    """The Inception case. `count_include_pad=True` is never written down.

    Reading node.args without applying schema defaults would see four
    arguments, conclude count_include_pad was absent, and detect nothing.
    """
    exported = export(DefaultedPool())
    pooling = [n for n in exported.graph.nodes
               if n.op == "call_function" and n.target in dd003_avgpool_pad.AVGPOOL_TARGETS]
    assert len(pooling[0].args) < 6, "test no longer exercises omitted defaults"
    assert dd003_avgpool_pad.detect(exported).applies


def test_ignores_a_graph_with_no_pooling():
    result = dd003_avgpool_pad.detect(export(NoPooling()))
    assert not result.applies
    assert result.skipped == []


def test_finds_every_site():
    assert len(dd003_avgpool_pad.detect(export(TwoSites())).detections) == 2


# --- the cases it must refuse -------------------------------------------------


def test_skips_count_include_pad_already_false():
    """Nothing to fix: XNNPACK accepts this node as it stands."""
    result = dd003_avgpool_pad.detect(export(Pool(count_include_pad=False)))
    assert not result.applies
    assert "already False" in result.skipped[0].reason


def test_skips_ceil_mode():
    """With ceil_mode a window may overhang the padded edge, shrinking the
    divisor in a way pre-padding cannot reproduce."""
    result = dd003_avgpool_pad.detect(
        export(Pool(ceil_mode=True), shape=(1, 8, 15, 15)))
    assert not result.applies
    assert "ceil_mode" in result.skipped[0].reason


def test_skips_divisor_override():
    result = dd003_avgpool_pad.detect(export(Pool(divisor_override=4)))
    assert not result.applies
    assert "divisor_override" in result.skipped[0].reason


def test_skips_dynamic_shapes():
    """A symbolic spatial dimension means the padding cannot be checked."""
    batch = torch.export.Dim("h", min=8, max=64)
    exported = torch.export.export(
        Pool().eval(), (torch.randn(1, 8, 16, 16),),
        dynamic_shapes={"x": {2: batch}})
    result = dd003_avgpool_pad.detect(exported)
    assert not result.applies
    assert "dynamic" in result.skipped[0].reason


def test_a_skipped_node_is_left_untouched():
    exported = export(Pool(ceil_mode=True), shape=(1, 8, 15, 15))
    before = copy.deepcopy(exported.graph_module.code)
    assert dd003_avgpool_pad.apply(exported) == 0
    assert exported.graph_module.code == before
    assert count_pads(exported) == 0


# --- the rewrite --------------------------------------------------------------


def test_apply_inserts_one_pad_and_clears_the_pooling_padding():
    exported = export(Pool())
    assert dd003_avgpool_pad.apply(exported) == 1
    assert count_pads(exported) == 1
    assert count_pools(exported) == 1

    node = [n for n in exported.graph.nodes
            if n.op == "call_function" and n.target in dd003_avgpool_pad.AVGPOOL_TARGETS][0]
    # Fully positional, so the partitioner reads what this rule reasoned about.
    assert node.args[3] == [0, 0], "padding must be gone from the operator"
    assert node.args[4] is False, "ceil_mode"
    assert node.args[5] is False, "count_include_pad"
    assert node.kwargs == {}


def test_zero_padding_needs_no_pad_node():
    """The DenseNet form. With nothing padded the flag is already a no-op, so
    flipping it is the entire repair."""
    exported = export(Pool(kernel_size=2, stride=2, padding=0))
    assert dd003_avgpool_pad.detect(exported).detections[0].needs_pad_node is False
    assert dd003_avgpool_pad.apply(exported) == 1
    assert count_pads(exported) == 0
    assert count_pools(exported) == 1


def test_apply_is_idempotent():
    """After the rewrite count_include_pad is False, so a second pass is a
    no-op rather than a second layer of padding."""
    exported = export(Pool())
    assert dd003_avgpool_pad.apply(exported) == 1
    assert dd003_avgpool_pad.apply(exported) == 0
    assert count_pads(exported) == 1


def test_the_rewritten_graph_still_runs():
    exported = export(Pool())
    dd003_avgpool_pad.apply(exported)
    exported.module()(torch.randn(1, 8, 16, 16))


def test_output_shape_is_unchanged():
    """Moving padding out of the operator does not move its boundaries."""
    probe = torch.randn(1, 8, 16, 16)
    exported = export(Pool())
    before = exported.module()(probe).shape
    dd003_avgpool_pad.apply(exported)
    assert exported.module()(probe).shape == before


def test_pad_node_carries_shape_metadata():
    """Downstream passes read meta['val']; a missing one breaks lowering."""
    exported = export(Pool())
    dd003_avgpool_pad.apply(exported)
    pad = [n for n in exported.graph.nodes
           if n.op == "call_function"
           and n.target is torch.ops.aten.constant_pad_nd.default][0]
    assert tuple(pad.meta["val"].shape) == (1, 8, 18, 18)


# --- equivalence, the claim the rule rests on ---------------------------------


EQUIVALENCE_CASES = [
    # shape,               kernel, stride, padding
    ((1, 8, 16, 16), 3, 1, 1),      # Inception
    ((1, 8, 56, 56), 2, 2, 0),      # DenseNet transition
    ((1, 3, 35, 35), 3, 1, 1),
    ((1, 4, 17, 17), 3, 2, 1),      # odd input, striding past the edge
    ((2, 5, 15, 15), 5, 1, 2),      # larger kernel, batch > 1
    ((1, 1, 4, 4), 2, 2, 1),
    ((1, 6, 5, 6), 3, 1, 1),        # non-square
    ((1, 2, 3, 3), 3, 1, 1),        # kernel covers the whole input
]


@pytest.mark.parametrize("shape,kernel,stride,padding", EQUIVALENCE_CASES)
def test_rewrite_is_bit_identical(shape, kernel, stride, padding):
    """Not "within tolerance" - equal.

    The padded positions contribute 0.0 to the sum in both forms, and with
    ceil_mode False both divide by the full pooling region.
    """
    torch.manual_seed(0)
    probe = torch.randn(*shape)
    exported = export(Pool(kernel, stride, padding), shape=shape)
    expected = exported.module()(probe)

    repaired = copy.deepcopy(exported)
    assert dd003_avgpool_pad.apply(repaired) == 1
    assert torch.equal(repaired.module()(probe), expected)


def test_rewrite_survives_extreme_values():
    """Padding zeros must not perturb a tensor whose real values are large."""
    probe = torch.full((1, 4, 8, 8), 1e6)
    exported = export(Pool(), shape=(1, 4, 8, 8))
    expected = exported.module()(probe)
    dd003_avgpool_pad.apply(exported)
    assert torch.equal(exported.module()(probe), expected)


# --- catalog wiring -----------------------------------------------------------


def test_registered_in_the_catalog():
    assert [rule.RULE_ID for rule in ALL_RULES] == ["DD-001", "DD-002", "DD-003"]


def test_matches_its_portable_kernel_and_no_other():
    assert dd003_avgpool_pad.matches_portable_kernel("native_call_avg_pool2d.out")
    assert not dd003_avgpool_pad.matches_portable_kernel("native_call__softmax.out")
    assert not dd003_avgpool_pad.matches_portable_kernel("native_call_alias_copy.out")


def test_rules_do_not_overlap():
    """DD-002 and DD-003 must not both claim the same node."""
    assert not dd002_noop_alias.matches_portable_kernel("native_call_avg_pool2d.out")
    exported = export(Pool())
    assert dd002_noop_alias.detect(exported).applies is False


def test_no_model_specific_special_casing():
    """The rule reasons about operator arguments, never about identities."""
    source = (dd003_avgpool_pad.detect.__doc__ or "") + (dd003_avgpool_pad.apply.__doc__ or "")
    import inspect
    body = inspect.getsource(dd003_avgpool_pad.detect) + inspect.getsource(dd003_avgpool_pad.apply)
    for forbidden in ("inception", "densenet", "avg_pool2d_3", "torchvision"):
        assert forbidden not in body.lower()
