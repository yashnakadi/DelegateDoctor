"""Tests for DD-002 - no-op alias removal.

Offline: torch.export only. No adb, device, NDK, network or native build.
"""

import copy

import pytest
import torch

from delegate_doctor.export_model import export_to_aten
from delegate_doctor.repairs import ALL_RULES, dd001_softmax, dd002_noop_alias


class FullSliceAlias(torch.nn.Module):
    """A slice that covers the whole tensor, as timm's GhostModule produces.

    `out[:, :C]` where C is the full channel count exports as `aten.alias`.
    """

    def __init__(self, channels=8):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 1)
        self.channels = channels

    def forward(self, x):
        out = self.conv(x)
        return out[:, : self.channels, :, :]


class PartialSlice(torch.nn.Module):
    """A genuine slice - must NOT be treated as an alias."""

    def __init__(self, channels=8):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return self.conv(x)[:, :4, :, :]


class NoAlias(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)


class TwoAliasSites(torch.nn.Module):
    def __init__(self, channels=8):
        super().__init__()
        self.a = torch.nn.Conv2d(channels, channels, 1)
        self.b = torch.nn.Conv2d(channels, channels, 1)
        self.channels = channels

    def forward(self, x):
        x = self.a(x)[:, : self.channels]
        x = self.b(x)[:, : self.channels]
        return x


def export(module, shape=(1, 8, 16, 16)):
    return export_to_aten(module.eval(), (torch.randn(*shape),))


def count_alias(exported):
    return sum(
        1
        for node in exported.graph.nodes
        if node.op == "call_function" and node.target in dd002_noop_alias.ALIAS_TARGETS
    )


# --- detection -------------------------------------------------------------

def test_full_width_slice_alias_is_detected():
    exported = export(FullSliceAlias())
    assert count_alias(exported) >= 1

    result = dd002_noop_alias.detect(exported)
    assert result.applies
    detection = result.detections[0]
    assert detection.shape == (1, 8, 16, 16)
    assert detection.dtype == "float32"
    assert "identity" in detection.explain()


def test_partial_slice_is_not_detected():
    """A real slice must be left alone - it is not an identity."""
    exported = export(PartialSlice())
    assert not dd002_noop_alias.detect(exported).applies


def test_graph_without_alias_is_not_detected():
    exported = export(NoAlias())
    result = dd002_noop_alias.detect(exported)
    assert not result.applies
    assert result.detections == []


def test_multiple_sites_are_all_found():
    exported = export(TwoAliasSites())
    assert len(dd002_noop_alias.detect(exported).detections) == count_alias(exported)


def test_dynamic_shape_is_declined_rather_than_guessed():
    """A symbolic dimension means the identity cannot be proven statically."""
    model = FullSliceAlias().eval()
    example = (torch.randn(2, 8, 16, 16),)
    batch = torch.export.Dim("batch", min=2, max=8)
    exported = torch.export.export(
        model, example, dynamic_shapes={"x": {0: batch}}, strict=True
    )

    result = dd002_noop_alias.detect(exported)
    if count_alias(exported):           # only meaningful if an alias survived
        assert not result.applies
        assert any("dynamic" in s.reason for s in result.skipped)


def test_shape_changing_node_would_be_skipped():
    """Guard the precondition directly: only shape+dtype identity is removed."""
    exported = export(FullSliceAlias())
    node = next(
        n for n in exported.graph.nodes
        if n.op == "call_function" and n.target in dd002_noop_alias.ALIAS_TARGETS
    )
    # Pretend the output shape differs from the input.
    node.meta["val"] = torch.empty(1, 4, 16, 16)
    result = dd002_noop_alias.detect(exported)
    assert not result.applies
    assert any("not an identity" in s.reason for s in result.skipped)


# --- rewrite ---------------------------------------------------------------

def test_removal_is_bit_exact():
    """Deleting a no-op must change nothing at all, not merely stay in tolerance."""
    torch.manual_seed(0)
    model = FullSliceAlias().eval()
    example = (torch.randn(1, 8, 16, 16),)
    with torch.no_grad():
        expected = model(*example)

    exported = export_to_aten(model, example)
    assert dd002_noop_alias.apply(exported) >= 1
    actual = exported.module()(*example)

    assert torch.equal(actual, expected)       # bit-identical
    assert tuple(actual.shape) == tuple(expected.shape)


def test_apply_removes_every_site_and_leaves_none_behind():
    exported = export(TwoAliasSites())
    before = count_alias(exported)
    assert dd002_noop_alias.apply(exported) == before
    assert count_alias(exported) == 0


def test_apply_is_a_no_op_when_nothing_is_detected():
    exported = export(NoAlias())
    assert dd002_noop_alias.apply(exported) == 0


def test_partial_slice_survives_the_repair():
    torch.manual_seed(0)
    model = PartialSlice().eval()
    example = (torch.randn(1, 8, 16, 16),)
    with torch.no_grad():
        expected = model(*example)
    exported = export_to_aten(model, example)
    dd002_noop_alias.apply(exported)
    assert torch.equal(exported.module()(*example), expected)


# --- rule plumbing ---------------------------------------------------------

def test_rule_identity_and_kernel_matching():
    assert dd002_noop_alias.RULE_ID == "DD-002"
    assert dd002_noop_alias.matches_portable_kernel("native_call_alias_copy.out")
    assert not dd002_noop_alias.matches_portable_kernel("native_call__softmax.out")
    # and the two rules claim disjoint kernels
    assert not dd001_softmax.matches_portable_kernel("native_call_alias_copy.out")


def test_both_rules_are_registered_in_order():
    assert [rule.RULE_ID for rule in ALL_RULES] == ["DD-001", "DD-002"]
    for rule in ALL_RULES:
        assert hasattr(rule, "detect") and hasattr(rule, "apply")
        assert hasattr(rule, "describe_rewrite")
        assert hasattr(rule, "matches_portable_kernel")


# --- interaction with DD-001 ----------------------------------------------

class SoftmaxAndAlias(torch.nn.Module):
    """Both patterns in one graph, as a real model could have."""

    def __init__(self, channels=8):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 1)
        self.channels = channels

    def forward(self, x):
        out = self.conv(x)[:, : self.channels, :, :]
        return torch.softmax(out, dim=1)


def test_the_two_rules_do_not_interfere():
    torch.manual_seed(0)
    model = SoftmaxAndAlias().eval()
    example = (torch.randn(1, 8, 16, 16),)
    with torch.no_grad():
        expected = model(*example)

    exported = export_to_aten(model, example)
    assert dd001_softmax.detect(exported).applies
    assert dd002_noop_alias.detect(exported).applies

    # applied in registry order, both should land
    assert dd001_softmax.apply(exported) == 1
    assert dd002_noop_alias.apply(exported) >= 1

    actual = exported.module()(*example)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0)


def test_dd002_does_not_touch_softmax_nodes():
    exported = export(SoftmaxAndAlias())
    softmax_before = sum(
        1 for n in exported.graph.nodes
        if n.op == "call_function" and n.target in dd001_softmax.SOFTMAX_TARGETS
    )
    dd002_noop_alias.apply(exported)
    softmax_after = sum(
        1 for n in exported.graph.nodes
        if n.op == "call_function" and n.target in dd001_softmax.SOFTMAX_TARGETS
    )
    assert softmax_before == softmax_after == 1
