"""Mapping measured runtime operators back to exported graph nodes.

The bug this file exists for: Swin-T measured three real hotspots on the Arm
target and AI repair rejected all of them with "the hotspot operator was not
found in the graph". The operators were in the graph. The names had changed,
because the runtime reports what it *executed* - after lowering, decomposition
and out-variant conversion - and the repair looked for that string in the
*exported* ATen graph.

    native_layer_norm.out   is not a substring of   aten::layer_norm
    expand_copy.out         is not a substring of   aten::expand

So the tests here are mostly about equivalence rather than equality, and about
refusing to guess when several nodes are equally good matches. Fully offline:
real `torch.export` graphs, no device, no provider.
"""

import pytest
import torch

from delegate_doctor import operator_correlation as correlation
from delegate_doctor import repair_loop
from delegate_doctor.agent import graph_context
from delegate_doctor.profiling import PortableKernel

# --- canonical form ------------------------------------------------------------


@pytest.mark.parametrize("runtime, exported", [
    # The three Swin-T hotspots that failed in the field.
    ("native_layer_norm.out", "aten::layer_norm"),
    ("native_layer_norm.out", "aten::native_layer_norm"),
    ("expand_copy.out", "aten::expand"),
    ("where.self_out", "aten::where"),
    # The systematic transformations behind them.
    ("_softmax.out", "aten::softmax"),
    ("add.out", "aten.add.Tensor"),
    ("slice_copy.Tensor_out", "aten::slice"),
    ("view_copy.default", "aten.view.default"),
    ("permute_copy.out", "aten::permute"),
    ("mean.out", "aten.mean.dim"),
])
def test_runtime_and_exported_names_canonicalize_alike(runtime, exported):
    assert correlation.canonical_operator(runtime) == \
        correlation.canonical_operator(exported)
    assert correlation.canonical_operator(runtime) != ""


@pytest.mark.parametrize("name, expected", [
    ("native_call__softmax.out", "softmax"),   # the profiling prefix
    ("aten::layer_norm", "layer_norm"),        # the :: namespace
    ("aten.expand.default", "expand"),         # the . namespace
    ("torch.ops.aten.where.self", "where"),    # the fully-qualified form
    ("native_layer_norm.out", "layer_norm"),   # native_ + .out
    ("expand_copy.out", "expand"),             # _copy + .out
    ("where.self_out", "where"),               # a compound overload
    ("_softmax", "softmax"),                   # the private spelling
])
def test_canonical_form_strips_each_kind_of_decoration(name, expected):
    assert correlation.canonical_operator(name) == expected


@pytest.mark.parametrize("name", ["", "   ", None])
def test_an_empty_name_canonicalizes_to_nothing(name):
    assert correlation.canonical_operator(name) == ""


def test_distinct_operators_do_not_collide():
    """Reduction must not be so aggressive that unrelated ops merge."""
    roots = {correlation.canonical_operator(name) for name in
             ("aten::layer_norm", "aten::batch_norm", "aten::group_norm",
              "aten::expand", "aten::view", "aten::where", "aten::softmax",
              "aten::mean", "aten::sum")}
    assert len(roots) == 9


def test_the_normalization_rules_live_in_one_place():
    """No module may do its own string surgery on an operator name."""
    import inspect
    from pathlib import Path

    package = Path(inspect.getfile(correlation)).parent
    allowed = {"operator_correlation.py"}
    for path in package.rglob("*.py"):
        if path.name in allowed:
            continue
        text = path.read_text()
        for token in ('.replace("_", "")', 'lstrip("_")',
                      '.rstrip("_out")', 'removesuffix(".out")'):
            assert token not in text, \
                f"{path.name} normalizes operator names itself: {token}"


# --- resolution against a real graph --------------------------------------------


class SwinLike(torch.nn.Module):
    """The Swin-T shape of the problem, in miniature.

    Two LayerNorms (so the operator is genuinely ambiguous by name alone), a
    `where`, and an `expand` - the three operators the field failure reported.
    """

    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(8)
        self.norm2 = torch.nn.LayerNorm(8)

    def forward(self, x, mask):
        x = self.norm1(x)
        x = x + torch.where(mask > 0, x, torch.zeros_like(x))
        x = self.norm2(x)
        return x + mask.expand(x.shape)


@pytest.fixture
def swin_like():
    return torch.export.export(
        SwinLike().eval(),
        (torch.randn(2, 4, 8), torch.randn(2, 4, 8)))


class SingleNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(8)

    def forward(self, x):
        return self.norm(x)


@pytest.fixture
def single_norm():
    return torch.export.export(SingleNorm().eval(), (torch.randn(2, 4, 8),))


def test_native_layer_norm_out_resolves_to_the_exported_layer_norm(single_norm):
    """The headline case: a decomposed out-variant finds its composite node."""
    resolution = correlation.resolve_hotspot(single_norm, "native_layer_norm.out")
    assert resolution.resolved
    assert "layer_norm" in resolution.node_id


def test_expand_copy_out_resolves_to_the_exported_expand(swin_like):
    resolution = correlation.resolve_hotspot(swin_like, "expand_copy.out")
    assert resolution.resolved
    assert "expand" in resolution.node_id


def test_where_self_out_resolves_to_the_exported_where(swin_like):
    resolution = correlation.resolve_hotspot(swin_like, "where.self_out")
    assert resolution.resolved
    assert "where" in resolution.node_id


def test_the_swin_t_failure_pattern_no_longer_reports_a_missing_operator(
        swin_like, single_norm):
    """The exact three hotspots from the field report, end to end.

    Each must correlate to a graph region. `native_layer_norm.out` is measured
    against a graph with one LayerNorm here; the two-LayerNorm case is
    ambiguity, tested separately, and is a different answer from "not found".
    """
    measured = [
        ("native_layer_norm.out", single_norm),
        ("expand_copy.out", swin_like),
        ("where.self_out", swin_like),
    ]
    for runtime_operator, program in measured:
        resolution = correlation.resolve_hotspot(program, runtime_operator)
        assert resolution.status != correlation.UNRESOLVED, (
            f"{runtime_operator} still reports as missing from the graph")
        assert resolution.resolved, f"{runtime_operator} did not resolve"

        # And the resolved node really does produce a graph neighbourhood, so
        # the request reaches the provider instead of stopping at NO_CANDIDATE.
        neighbourhood = graph_context.build_neighbourhood(
            program, runtime_operator, node_name=resolution.node_id)
        assert neighbourhood.nodes, f"{runtime_operator} built no graph context"
        assert neighbourhood.hotspot_identifier


def test_an_unrelated_runtime_operator_fails_cleanly(swin_like):
    resolution = correlation.resolve_hotspot(swin_like, "definitely_not_an_op.out")
    assert resolution.status == correlation.UNRESOLVED
    assert not resolution.resolved
    assert resolution.node_id == ""
    assert "could not be correlated" in resolution.reason


# --- ambiguity -------------------------------------------------------------------


def test_several_matching_nodes_are_not_resolved_arbitrarily(swin_like):
    """Two LayerNorms and no way to tell them apart: refuse, do not pick."""
    resolution = correlation.resolve_hotspot(swin_like, "native_layer_norm.out")
    assert resolution.status == correlation.AMBIGUOUS
    assert not resolution.resolved
    assert resolution.candidate_count == 2
    assert "2 possible graph nodes" in resolution.reason


def test_the_ambiguous_message_differs_from_the_missing_message(swin_like):
    """"Found none" and "found several" must not read the same."""
    ambiguous = correlation.resolve_hotspot(swin_like, "native_layer_norm.out")
    missing = correlation.resolve_hotspot(swin_like, "no_such_operator.out")
    assert ambiguous.reason != missing.reason
    assert "uniquely" in ambiguous.reason
    assert "could not be correlated" in missing.reason


def test_a_matching_site_count_disambiguates_by_execution_order(swin_like):
    """Two measured sites and two candidate nodes is a correspondence."""
    first = correlation.resolve_hotspot(
        swin_like, "native_layer_norm.out", occurrence=0, site_count=2)
    second = correlation.resolve_hotspot(
        swin_like, "native_layer_norm.out", occurrence=1, site_count=2)
    assert first.resolved and second.resolved
    assert first.node_id != second.node_id


def test_a_mismatched_site_count_does_not_disambiguate(swin_like):
    """Three measured sites against two nodes is not a correspondence."""
    resolution = correlation.resolve_hotspot(
        swin_like, "native_layer_norm.out", occurrence=0, site_count=3)
    assert resolution.status == correlation.AMBIGUOUS


def test_a_debug_node_id_wins_over_every_name_comparison(swin_like):
    """Stable identity from lowering is used directly when it exists."""
    resolution = correlation.resolve_hotspot(
        swin_like, "native_layer_norm.out", debug_node_id="layer_norm_1")
    assert resolution.resolved
    assert resolution.node_id == "layer_norm_1"
    assert "debug metadata" in resolution.detail


def test_a_claimed_node_is_not_handed_to_a_second_hotspot(swin_like):
    """Two hotspots must not both target the same node."""
    first = correlation.resolve_hotspot(swin_like, "where.self_out")
    second = correlation.resolve_hotspot(
        swin_like, "where.self_out", exclude=frozenset({first.node_id}))
    assert first.resolved
    assert not second.resolved


# --- multiple occurrences, through the hotspot layer --------------------------------


def kernel(operator, total_ms, fraction, site_costs=(), debug=""):
    return PortableKernel(name=f"native_call_{operator}", total_ms=total_ms,
                          call_count=max(1, len(site_costs)),
                          runtime_fraction=fraction, site_costs=site_costs,
                          debug_node_id=debug)


class FakeProfile:
    def __init__(self, kernels):
        self.portable_kernels = list(kernels)
        self.method_execute_ms = 100.0
        self.portable_ms = sum(item.total_ms for item in kernels)
        self.delegated_ms = 100.0 - self.portable_ms
        self.delegate_call_count = 1
        self.operator_call_count = len(kernels)
        self.accounting_warning = ""

    @property
    def runtime_delegation_fraction(self):
        return 1.0 - sum(item.runtime_fraction for item in self.portable_kernels)


def test_two_layer_norm_sites_become_two_hotspots_with_distinct_nodes(swin_like):
    """The aggregate is one operator; the repair targets are two nodes."""
    profile = FakeProfile([kernel("native_layer_norm.out", 10.0, 0.10,
                                  site_costs=(7.0, 3.0))])
    hotspots = repair_loop.collect_hotspots(profile, swin_like)

    assert len(hotspots) == 2
    assert len({item.node_id for item in hotspots}) == 2
    assert all(item.targetable for item in hotspots)
    # Highest-cost site first: that is the one worth repairing.
    assert hotspots[0].event_time_ms == 7.0
    assert hotspots[0].runtime_share > hotspots[1].runtime_share


def test_the_aggregate_operator_share_is_split_across_its_sites(swin_like):
    profile = FakeProfile([kernel("native_layer_norm.out", 10.0, 0.10,
                                  site_costs=(7.0, 3.0))])
    hotspots = repair_loop.collect_hotspots(profile, swin_like)
    assert sum(item.runtime_share for item in hotspots) == pytest.approx(0.10)
    assert hotspots[0].runtime_share == pytest.approx(0.07)


def test_the_profile_still_reports_the_operator_total(swin_like):
    """Reporting aggregates; targeting does not. Both stay available."""
    profile = FakeProfile([kernel("native_layer_norm.out", 10.0, 0.10,
                                  site_costs=(7.0, 3.0))])
    assert profile.portable_kernels[0].total_ms == 10.0
    assert profile.portable_kernels[0].runtime_fraction == 0.10
    assert profile.portable_kernels[0].site_count == 2


def test_an_ambiguous_hotspot_is_measured_but_not_targetable(swin_like):
    """No per-site breakdown and two candidates: recorded, not repairable."""
    profile = FakeProfile([kernel("native_layer_norm.out", 10.0, 0.10)])
    hotspots = repair_loop.collect_hotspots(profile, swin_like)
    assert len(hotspots) == 1
    assert not hotspots[0].targetable
    assert hotspots[0].resolution_status == correlation.AMBIGUOUS
    assert "uniquely" in hotspots[0].resolution_reason


def test_the_swin_hotspots_become_targetable_hotspots(swin_like):
    """The field failure, at the layer the repair loop actually consumes."""
    profile = FakeProfile([
        kernel("native_layer_norm.out", 10.1, 0.101, site_costs=(6.0, 4.1)),
        kernel("expand_copy.out", 5.7, 0.057),
        kernel("where.self_out", 1.5, 0.015),
    ])
    hotspots = repair_loop.collect_hotspots(profile, swin_like)
    by_operator = {}
    for item in hotspots:
        by_operator.setdefault(item.operator_name, []).append(item)

    assert len(by_operator["native_layer_norm.out"]) == 2
    assert all(item.targetable for group in by_operator.values()
               for item in group), \
        "a measured Swin-T hotspot could not be targeted"


def test_a_debug_handle_carries_through_to_the_hotspot(swin_like):
    profile = FakeProfile([kernel("native_layer_norm.out", 10.0, 0.10,
                                  debug="layer_norm_1")])
    hotspots = repair_loop.collect_hotspots(profile, swin_like)
    assert hotspots[0].node_id == "layer_norm_1"


def test_hotspots_without_a_graph_are_reported_untargetable():
    """No program to correlate against: measured, never rewritten."""
    profile = FakeProfile([kernel("native_layer_norm.out", 10.0, 0.10)])
    hotspots = repair_loop.collect_hotspots(profile, None)
    assert len(hotspots) == 1
    assert not hotspots[0].targetable


# --- graph context uses identity, not a name ------------------------------------------


def test_graph_context_centres_on_the_named_node(swin_like):
    """The third LayerNorm's neighbourhood is not the first one's."""
    first = graph_context.build_neighbourhood(
        swin_like, "native_layer_norm.out", node_name="layer_norm")
    second = graph_context.build_neighbourhood(
        swin_like, "native_layer_norm.out", node_name="layer_norm_1")
    assert first.hotspot_identifier != second.hotspot_identifier


def test_graph_context_refuses_to_guess_between_several_nodes(swin_like):
    """Without a node name and with two candidates, describe neither."""
    neighbourhood = graph_context.build_neighbourhood(
        swin_like, "native_layer_norm.out")
    assert neighbourhood.nodes == []


def test_graph_context_resolves_an_unambiguous_operator_by_name(swin_like):
    neighbourhood = graph_context.build_neighbourhood(
        swin_like, "expand_copy.out")
    assert neighbourhood.nodes
    assert "expand" in neighbourhood.hotspot_operator


def test_graph_context_still_sends_only_approved_metadata(swin_like):
    """Correlation changed what is targeted, not what is transmitted."""
    neighbourhood = graph_context.build_neighbourhood(
        swin_like, "where.self_out")
    payload = neighbourhood.to_dict()
    text = str(payload)
    for forbidden in ("0x", "state_dict", "weight_data", "Parameter containing"):
        assert forbidden not in text, forbidden
    for node in payload["nodes"]:
        assert set(node) <= {"id", "op", "target", "inputs", "literals",
                             "shape", "dtype", "hotspot"}


# --- diagnostics ------------------------------------------------------------------------


def test_the_failure_message_is_terse_by_default(swin_like):
    resolution = correlation.resolve_hotspot(swin_like, "native_layer_norm.out")
    terse = correlation.format_resolution_failure(resolution, verbose=False)
    assert "2 possible graph nodes" in terse
    assert "layer_norm_1" not in terse, "graph internals leaked into a normal run"


def test_the_failure_message_carries_detail_when_verbose(swin_like):
    resolution = correlation.resolve_hotspot(swin_like, "native_layer_norm.out")
    detailed = correlation.format_resolution_failure(resolution, verbose=True)
    assert "candidates:" in detailed
    assert "layer_norm" in detailed


def test_a_resolution_survives_the_json_round_trip(swin_like):
    import json

    resolution = correlation.resolve_hotspot(swin_like, "expand_copy.out")
    payload = json.loads(json.dumps(resolution.to_dict()))
    assert payload["status"] == correlation.RESOLVED
    assert payload["canonical"] == "expand"


# --- the catalog is unaffected --------------------------------------------------------


def test_catalog_matching_still_works_on_the_runtime_kernel_name():
    """DD rules match the ETDump kernel name, and that has not changed."""
    from delegate_doctor.repairs import ALL_RULES

    lookup = repair_loop.catalog_lookup_for(ALL_RULES)
    assert lookup("native_call__softmax.out") == "DD-001"
    assert lookup("native_call_native_layer_norm.out") is None


def test_a_catalog_hotspot_is_still_collected_when_it_cannot_be_located():
    """A known repair does not need a resolved node: the rule finds its own."""
    profile = FakeProfile([kernel("_softmax.out", 10.0, 0.10)])
    hotspots = repair_loop.collect_hotspots(
        profile, None,
        catalog_lookup=lambda name: "DD-001" if "softmax" in name else None)
    assert hotspots[0].catalog_match == "DD-001"
    assert not hotspots[0].targetable
