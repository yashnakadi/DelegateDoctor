"""A catalog rule is scheduled once per graph state, not once per hotspot.

The bug this file pins down: DD-003 matches nine `avg_pool2d` sites in
Inception V3, and the loop scheduled it nine times. Every attempt called the
same deterministic `apply()`, which rewrites every site it recognises, so all
nine produced an identical candidate, met identical gates, and were rejected
identically - nine report rows for one decision, and nine device benchmarks.

Fully offline: real `torch.export` graphs, mocked device gates, no network.
"""

import torch
import torch.nn.functional as F

from delegate_doctor import pipeline, repair_loop
from delegate_doctor.profiling import PortableKernel, ProfileResult
from delegate_doctor.repairs import ALL_RULES, dd003_avgpool_pad

from tests.test_repair_loop import (  # noqa: F401
    Profiles, gates, no_provider, run, spec_for)


# --- models ---------------------------------------------------------------------

class NinePools(torch.nn.Module):
    """Inception-shaped: nine identical padded pooling sites, one rule."""

    def forward(self, x):
        for _ in range(9):
            x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x


class OnePool(torch.nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)


class PoolAndSoftmax(torch.nn.Module):
    """Two operators, two different rules, one graph."""

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return torch.softmax(x, dim=1)


def pool_spec(model=None):
    return spec_for(model or NinePools(), (torch.randn(1, 4, 8, 8),))


def multi_site_profile(operator="avg_pool2d.out", share=0.609, sites=9,
                       total_ms=100.0):
    """One kernel measured at `sites` separate instruction sites.

    Costs are deliberately unequal, the way a real trace is: the nine Inception
    sites ranged from 10.9% down to 2.4%.
    """
    weights = [sites - index for index in range(sites)]
    scale = share * total_ms / sum(weights)
    site_costs = tuple(weight * scale for weight in weights)
    kernel = PortableKernel(
        name=f"native_call_{operator}", total_ms=sum(site_costs),
        call_count=sites, runtime_fraction=share, site_costs=site_costs)
    return ProfileResult(
        method_execute_ms=total_ms,
        delegated_ms=total_ms - kernel.total_ms,
        portable_ms=kernel.total_ms,
        delegate_call_count=1,
        operator_call_count=sites + 1,
        portable_kernels=[kernel],
    )


def two_operator_profile():
    kernels = [
        PortableKernel(name="native_call_avg_pool2d.out", total_ms=40.0,
                       call_count=1, runtime_fraction=0.40),
        PortableKernel(name="native_call__softmax.out", total_ms=30.0,
                       call_count=1, runtime_fraction=0.30),
    ]
    return ProfileResult(
        method_execute_ms=100.0, delegated_ms=30.0, portable_ms=70.0,
        delegate_call_count=1, operator_call_count=3,
        portable_kernels=kernels)


def dd003_attempts(history):
    return [attempt for attempt in history.attempts
            if attempt.repair_id == "DD-003"]


# --- grouping, as a unit ----------------------------------------------------------

def hotspots_for(profile, program):
    return repair_loop.collect_hotspots(
        profile, program, repair_loop.catalog_lookup_for(ALL_RULES))


def test_nine_matching_sites_become_one_rule_match():
    spec = pool_spec()
    hotspots = hotspots_for(multi_site_profile(), spec.exported_program)
    assert len(hotspots) == 9, "the profile really does describe nine sites"

    matches = repair_loop.group_catalog_matches(hotspots)
    assert len(matches) == 1
    assert matches[0].rule_id == "DD-003"
    assert matches[0].measured_site_count == 9


def test_the_match_aggregates_the_sites_runtime_without_double_counting():
    spec = pool_spec()
    hotspots = hotspots_for(multi_site_profile(share=0.609),
                            spec.exported_program)
    match = repair_loop.group_catalog_matches(hotspots)[0]

    assert match.runtime_share == \
        __import__("pytest").approx(0.609, abs=1e-9)
    # Not the largest single site, and not a sum that ran away.
    assert match.runtime_share > max(h.runtime_share for h in hotspots)
    assert match.runtime_share <= 1.0


def test_an_aggregate_can_never_exceed_all_of_runtime():
    """A share above 100% would be a claim no measurement can support."""
    spec = pool_spec()
    hotspots = hotspots_for(multi_site_profile(share=0.99),
                            spec.exported_program)
    inflated = repair_loop.CatalogRuleMatch(
        rule_id="DD-003", hotspots=tuple(hotspots) * 3)
    assert inflated.runtime_share == 1.0


def test_two_rules_on_one_graph_are_two_independent_matches():
    spec = pool_spec(PoolAndSoftmax())
    hotspots = hotspots_for(two_operator_profile(), spec.exported_program)
    matches = repair_loop.group_catalog_matches(hotspots)

    assert {match.rule_id for match in matches} == {"DD-001", "DD-003"}
    # Ordered by aggregate cost, so the more expensive rule goes first.
    assert matches[0].rule_id == "DD-003"


def test_the_costliest_rule_is_scheduled_first():
    spec = pool_spec(PoolAndSoftmax())
    hotspots = hotspots_for(two_operator_profile(), spec.exported_program)
    match = repair_loop.next_catalog_match(hotspots, frozenset(), "fp")
    assert match.rule_id == "DD-003"


# --- one attempt per rule per graph state -----------------------------------------

def test_an_attempted_rule_is_not_offered_again_on_the_same_graph():
    spec = pool_spec()
    hotspots = hotspots_for(multi_site_profile(), spec.exported_program)
    attempted = {repair_loop.attempt_key("fp-1", "DD-003")}

    assert repair_loop.next_catalog_match(hotspots, attempted, "fp-1") is None


def test_the_same_rule_is_offered_again_once_the_graph_changes():
    """An accepted repair produces a new graph, where more work may remain."""
    spec = pool_spec()
    hotspots = hotspots_for(multi_site_profile(), spec.exported_program)
    attempted = {repair_loop.attempt_key("fp-1", "DD-003")}

    match = repair_loop.next_catalog_match(hotspots, attempted, "fp-2")
    assert match is not None and match.rule_id == "DD-003"


def test_the_fingerprint_tracks_the_graph_contents():
    import copy

    spec = pool_spec()
    before = repair_loop.graph_fingerprint(spec.exported_program)
    assert before == repair_loop.graph_fingerprint(spec.exported_program)

    rewritten = copy.deepcopy(spec.exported_program)
    dd003_avgpool_pad.apply(rewritten)
    assert repair_loop.graph_fingerprint(rewritten) != before


def test_an_unfingerprintable_graph_degrades_to_attempting_once():
    """Conservative direction: one attempt, never a retry loop."""
    class Unprintable:
        @property
        def graph(self):
            raise RuntimeError("no graph here")

    assert repair_loop.graph_fingerprint(Unprintable()) == ""


# --- end to end -------------------------------------------------------------------

def test_nine_sites_produce_exactly_one_rejected_attempt(tmp_path, monkeypatch,
                                                         gates):
    """The reported bug, as a regression test."""
    no_provider(monkeypatch)
    gates.latencies = [(50.0, 100.0)]          # slower, so the repair is rejected
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(multi_site_profile()))

    outcome = run(pool_spec(), tmp_path)
    history = outcome.repair_history

    assert len(dd003_attempts(history)) == 1, "one rule, one decision"
    assert history.rejected_count == 1
    assert len(gates.benchmarks) == 1, "one candidate, one benchmark"


def test_the_single_attempt_speaks_for_every_site(tmp_path, monkeypatch, gates):
    no_provider(monkeypatch)
    gates.latencies = [(100.0, 50.0)]
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(multi_site_profile(), multi_site_profile(share=0.0)))

    attempt = dd003_attempts(run(pool_spec(), tmp_path).repair_history)[0]

    assert attempt.matching_sites == 9, "apply() rewrote all nine"
    assert attempt.measured_sites == 9
    assert 0.60 < attempt.represented_runtime < 0.62


def test_the_terminal_names_the_rule_once(tmp_path, monkeypatch, gates, capsys):
    """Progress lines go to the terminal, so read them there."""
    no_provider(monkeypatch)
    gates.latencies = [(50.0, 100.0)]
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(multi_site_profile()))

    run(pool_spec(), tmp_path, quiet=False)
    text = capsys.readouterr().out

    assert text.count("DD-003 found") == 1
    assert "Applying to 9 matching sites..." in text
    # One numbered opener carrying the sites' aggregate cost, not nine.
    assert text.count("] avg_pool2d.out") == 1
    assert "[1] avg_pool2d.out" in text and "[2]" not in text
    assert "60.9%" in text


def test_verbose_keeps_the_per_site_breakdown(tmp_path, monkeypatch, gates,
                                              capsys):
    no_provider(monkeypatch)
    gates.latencies = [(50.0, 100.0)]
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(multi_site_profile()))

    run(pool_spec(), tmp_path, quiet=False, verbose=True)
    text = capsys.readouterr().out

    assert text.count("site 1:") == 1
    assert text.count("site 9:") == 1


def test_the_history_holds_one_record_for_the_rule(tmp_path, monkeypatch, gates):
    no_provider(monkeypatch)
    gates.latencies = [(50.0, 100.0)]
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(multi_site_profile()))

    history = run(pool_spec(), tmp_path).repair_history.to_dict()
    records = [a for a in history["attempts"] if a["repair_id"] == "DD-003"]

    assert len(records) == 1
    assert records[0]["matching_sites"] == 9
    assert records[0]["measured_sites"] == 9
    assert 0.60 < records[0]["represented_runtime"] < 0.62


def test_a_single_site_rule_still_reports_one_site(tmp_path, monkeypatch, gates):
    """DD-001 and DD-002 shaped runs are unchanged by rule-level scheduling."""
    no_provider(monkeypatch)
    gates.latencies = [(50.0, 100.0)]
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(multi_site_profile(sites=1, share=0.40)))

    attempt = dd003_attempts(run(pool_spec(OnePool()), tmp_path).repair_history)[0]
    assert attempt.matching_sites == 1
    assert attempt.measured_sites == 1


def test_two_rules_are_each_attempted_once(tmp_path, monkeypatch, gates):
    no_provider(monkeypatch)
    gates.latencies = [(50.0, 100.0)]
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(two_operator_profile(), two_operator_profile()))

    history = run(pool_spec(PoolAndSoftmax()), tmp_path).repair_history
    attempted = [attempt.repair_id for attempt in history.attempts
                 if attempt.repair_id]

    assert sorted(attempted) == ["DD-001", "DD-003"]


def test_an_accepted_repair_triggers_a_reprofile(tmp_path, monkeypatch, gates):
    no_provider(monkeypatch)
    gates.latencies = [(100.0, 50.0)]
    profiles = Profiles(multi_site_profile(), multi_site_profile(share=0.0))
    monkeypatch.setattr(pipeline.profiling, "profile_model", profiles)

    history = run(pool_spec(), tmp_path).repair_history

    assert history.accepted_count == 1
    # Original profile, plus one after the accepted repair.
    assert profiles.calls >= 2


# --- gate ordering ----------------------------------------------------------------

def test_a_host_failure_skips_the_device_work(tmp_path, monkeypatch, gates):
    """Measuring how fast a wrong answer arrives is device time wasted."""
    no_provider(monkeypatch)
    gates.host_passes = False
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(multi_site_profile()))

    history = run(pool_spec(), tmp_path).repair_history

    assert history.rejected_count == 1
    assert gates.benchmarks == [], "benchmarked a semantically invalid candidate"
    assert gates.device_calls == [], "verified a semantically invalid candidate"


def test_a_device_failure_skips_the_benchmark(tmp_path, monkeypatch, gates):
    no_provider(monkeypatch)
    gates.device_passes = False
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(multi_site_profile()))

    history = run(pool_spec(), tmp_path).repair_history

    assert history.rejected_count == 1
    assert gates.device_calls, "the device gate should still have run"
    assert gates.benchmarks == []


def test_a_host_failure_is_reported_as_a_host_failure(tmp_path, monkeypatch,
                                                      gates):
    no_provider(monkeypatch)
    gates.host_passes = False
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(multi_site_profile()))

    attempt = dd003_attempts(run(pool_spec(), tmp_path).repair_history)[0]
    assert attempt.host_verification_passed is False
    assert "host numerical verification failed" in attempt.reason
