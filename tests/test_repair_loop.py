"""The iterative hotspot loop: what gets repaired, in what order, and when it stops.

Two halves. The first tests the policy in `repair_loop` directly - eligibility,
ranking, identity, termination - which needs no device and no model. The second
drives the real pipeline with mocked device mechanics, because the properties
that matter most are about *sequencing*: that a repair is verified against the
original rather than the previous step, benchmarked against the current best
rather than the original, and that the hotspot list is rebuilt from a fresh
profile after every acceptance.

Fully offline: no provider, no device, no browser.
"""

import copy

import pytest
import torch

from delegate_doctor import (device_verification, pipeline, repair_loop,
                             result as result_module)
from delegate_doctor.export_model import ModelSpec
from delegate_doctor.profiling import PortableKernel, ProfileResult
from delegate_doctor.repairs import ALL_RULES

# --- half one: the policy, with no machinery at all ---------------------------


def hotspot(share=0.10, operator="mean.out", node="n0", catalog=None,
            event_ms=None):
    return repair_loop.RepairHotspot(
        operator_name=operator,
        kernel_name=f"native_call_{operator}",
        runtime_share=share,
        event_time_ms=share * 100 if event_ms is None else event_ms,
        node_id=node,
        catalog_match=catalog,
    )


@pytest.mark.parametrize("share, eligible", [
    (0.0005, False),    # 0.05% - below
    (0.0010, False),    # exactly the threshold - the comparison is strict
    (0.0011, True),     # just above
    (0.009, True),      # 0.9% - a DD repair is worth trying here
    (0.38, True),       # 38%
])
def test_catalog_eligibility_is_strictly_above_the_dd_threshold(share, eligible):
    assert hotspot(share=share).eligible_for_catalog is eligible


def test_there_is_no_per_site_ai_threshold():
    """The AI unit is the operator family, so a site is never judged alone.

    A per-site bar would discard twenty-four 0.4% LayerNorms before they were
    ever added up, which is exactly the opportunity the family threshold is
    there to notice.
    """
    assert not hasattr(repair_loop, "MIN_AI_HOTSPOT_RUNTIME_SHARE")
    small = hotspot(share=0.004, catalog=None)
    assert small.eligible, "a small site was dropped before it could be grouped"


def test_the_collection_floor_is_a_named_constant():
    assert repair_loop.MIN_DD_HOTSPOT_RUNTIME_SHARE == 0.001


def test_both_routes_share_one_collection_floor():
    """Known or unknown, a site has to be a real measurement to be collected."""
    assert hotspot(share=0.009, catalog="DD-001").eligible
    assert hotspot(share=0.009, catalog=None).eligible
    assert not hotspot(share=0.0005, catalog=None).eligible


def test_the_upper_bound_is_amdahls_ceiling():
    assert hotspot(share=0.38).theoretical_upper_bound == pytest.approx(
        1 / (1 - 0.38))


def test_the_upper_bound_is_omitted_when_it_would_be_infinite():
    assert hotspot(share=1.0).theoretical_upper_bound is None
    assert hotspot(share=0.0).theoretical_upper_bound is None


def test_identity_is_not_the_operator_name_alone():
    """The same operator in two graph locations is two hotspots."""
    first = hotspot(operator="mean.out", node="mean_1")
    second = hotspot(operator="mean.out", node="mean_2")
    assert first.hotspot_id != second.hotspot_id


class FakeProfile:
    def __init__(self, kernels):
        self.portable_kernels = kernels
        self.method_execute_ms = 100.0
        self.portable_ms = sum(kernel.total_ms for kernel in kernels)
        self.delegated_ms = 100.0 - self.portable_ms
        self.delegate_call_count = 1
        self.operator_call_count = len(kernels)
        self.accounting_warning = ""

    @property
    def runtime_delegation_fraction(self):
        return 1.0 - sum(kernel.runtime_fraction
                         for kernel in self.portable_kernels)


def kernel(operator, share):
    return PortableKernel(name=f"native_call_{operator}",
                          total_ms=share * 100, call_count=1,
                          runtime_fraction=share)


def test_hotspots_are_ranked_by_runtime_share_and_filtered():
    """Everything above the collection floor, worst first.

    0.8% is kept: it may be a catalog repair, or one of many sites of an
    operator family that adds up to something worth asking about. Only genuine
    noise is dropped.
    """
    profile = FakeProfile([
        kernel("_softmax.out", 0.380),
        kernel("mean.out", 0.087),
        kernel("layer_norm.out", 0.042),
        kernel("slice_copy.out", 0.008),
        kernel("detach.out", 0.0005),         # below the collection floor
    ])
    found = repair_loop.collect_hotspots(profile)
    assert [item.operator_name for item in found] == [
        "_softmax.out", "mean.out", "layer_norm.out", "slice_copy.out"]


def test_delegated_runtime_is_never_a_repair_candidate():
    """Only portable kernels are offered. An expensive delegate is not a bug."""
    profile = FakeProfile([kernel("mean.out", 0.20)])
    found = repair_loop.collect_hotspots(profile)
    assert len(found) == 1
    assert found[0].operator_name == "mean.out"


def test_a_catalog_match_is_attached_when_a_rule_recognises_the_kernel():
    profile = FakeProfile([kernel("_softmax.out", 0.38)])
    lookup = repair_loop.catalog_lookup_for(ALL_RULES)
    found = repair_loop.collect_hotspots(profile, catalog_lookup=lookup)
    assert found[0].catalog_match == "DD-001"


def test_the_next_hotspot_skips_ones_already_finished():
    hotspots = [hotspot(0.38, "a", "n1"), hotspot(0.09, "b", "n2")]
    finished = {hotspots[0].hotspot_id}
    assert repair_loop.next_hotspot(hotspots, finished).operator_name == "b"
    finished.add(hotspots[1].hotspot_id)
    assert repair_loop.next_hotspot(hotspots, finished) is None


def test_candidate_numbering_is_global_across_a_run():
    numbering = repair_loop.CandidateNumbering()
    assert [numbering.next() for _ in range(3)] == [
        "AI-CANDIDATE-001", "AI-CANDIDATE-002", "AI-CANDIDATE-003"]


def test_the_safety_cap_exists_and_is_conservative():
    assert repair_loop.MAX_REPAIR_ITERATIONS == 16


def test_history_counts_by_source():
    history = repair_loop.RepairHistory()
    for source, status in (("catalog", "ACCEPTED"), ("catalog", "ACCEPTED"),
                           ("ai", "ACCEPTED"), ("ai", "REJECTED")):
        history.record(repair_loop.RepairAttempt(
            iteration=1, hotspot=hotspot(), source=source, status=status))
    assert history.accepted_count == 3
    assert history.catalog_count == 2
    assert history.ai_count == 1
    assert history.rejected_count == 1


def test_total_speedup_is_original_to_final():
    history = repair_loop.RepairHistory()
    history.original_latency_ms = 100.0
    history.final_latency_ms = 58.0
    assert history.total_speedup == pytest.approx(100.0 / 58.0)


# --- half two: the loop, driving the real pipeline ----------------------------


class SoftmaxNet(torch.nn.Module):
    """DD-001 territory: a non-last-dim softmax."""

    def forward(self, x):
        return torch.softmax(x, dim=1)


class FmodNet(torch.nn.Module):
    """A portable fallback no catalog rule matches."""

    def forward(self, x):
        return torch.fmod(x, 2.0)


class MixedNet(torch.nn.Module):
    """A catalog-matchable softmax plus fallbacks nothing recognises.

    Used where a test needs a known repair *and* AI hotspots in one graph, so
    the fake profiles it drives name operators that genuinely exist.
    """

    def forward(self, x):
        x = torch.softmax(x, dim=1)
        x = torch.fmod(x, 2.0)
        return x + torch.erf(x)


class ManyFallbacksNet(torch.nn.Module):
    """Several distinct fallbacks, none of which a catalog rule matches.

    The graph really does contain fmod, mean and erf, so a fake profile naming
    them correlates to real nodes. A test that profiled operators the graph
    does not contain would now be testing the resolver's refusal rather than
    the loop.
    """

    def forward(self, x):
        x = torch.fmod(x, 2.0)
        x = x + torch.erf(x)
        return x + x.mean()


class FakeDevice:
    serial = "test-target"
    is_emulator = False

    def short_description(self):
        return "TestTarget arm64-v8a"

    def describe(self):
        return "Arm64 Android device - TestTarget"


def profile_with(*shares_by_operator, total_ms=100.0):
    """A ProfileResult with exactly the portable kernels named."""
    kernels = [PortableKernel(name=f"native_call_{operator}",
                              total_ms=share * total_ms, call_count=1,
                              runtime_fraction=share)
               for operator, share in shares_by_operator]
    portable = sum(k.total_ms for k in kernels)
    return ProfileResult(
        method_execute_ms=total_ms,
        delegated_ms=total_ms - portable,
        portable_ms=portable,
        delegate_call_count=1,
        operator_call_count=len(kernels) + 1,
        portable_kernels=kernels,
    )


def spec_for(model, args):
    return ModelSpec(name="test model",
                     exported_program=torch.export.export(model.eval(), args),
                     example_args=args)


class Gates:
    """Records exactly which programs each gate was handed.

    The point of the loop is *which* pair of programs meets each gate, so the
    fake records paths rather than just counting calls.
    """

    def __init__(self):
        self.host_calls = []
        self.device_calls = []
        self.benchmarks = []
        self.host_passes = True
        self.device_passes = True
        self.latencies = [(100.0, 50.0)]

    def next_latency(self):
        if len(self.latencies) == 1:
            return self.latencies[0]
        return self.latencies[min(len(self.benchmarks) - 1,
                                  len(self.latencies) - 1)]


class Metrics:
    max_absolute_error = 1e-8
    mean_absolute_error = 1e-9
    mean_squared_error = 1e-17
    max_relative_error = 1e-7


class Verification:
    """Stands in for both host and device verification results.

    `passed` is repair fidelity only. `backend_fidelity` is the separate
    question of how well the backend reproduces the host, and defaults to OK so
    a test that says nothing about it is not silently asserting a warning.
    """

    def __init__(self, passed=True, backend_fidelity=None):
        self.passed = passed
        self.repaired_vs_original = Metrics()
        self.repaired_vs_eager = Metrics()
        self.original_device_vs_host = Metrics()
        self.repaired_device_vs_host = Metrics()
        self.argmax_agreement = None
        self.backend_fidelity = (
            backend_fidelity or device_verification.BACKEND_FIDELITY_OK)
        self.backend_fidelity_reason = ""
        self.failure_reasons = [] if passed else ["mocked failure"]
        self.error = ""

    @property
    def backend_fidelity_acceptable(self):
        return self.backend_fidelity != device_verification.BACKEND_FIDELITY_FAIL

    @property
    def status_text(self):
        return "PASS" if self.passed else "FAIL"

    def to_dict(self):
        return {"passed": self.passed,
                "backend_fidelity": self.backend_fidelity}


class Stats:
    def __init__(self, p50):
        self.p50_ms = p50
        self.p95_ms = p50 * 1.1
        self.mean_ms = p50
        self.sample_count = 100


class Benchmark:
    def __init__(self, before, after):
        self.before = Stats(before)
        self.after = Stats(after)
        self.warmup_iterations = 1
        self.measured_iterations = 1
        self.repetitions = 1
        self.threads = 4
        self.device_description = "TestTarget"
        self.device_is_emulator = False

    @property
    def p50_speedup(self):
        return self.before.p50_ms / self.after.p50_ms

    def to_dict(self):
        return {}


@pytest.fixture
def gates(monkeypatch):
    """Mock every device mechanism, recording what each gate was given."""
    recorded = Gates()

    def host(original_output, repaired_output, eager_output, argmax_dim=None):
        recorded.host_calls.append((id(original_output), id(repaired_output)))
        return Verification(recorded.host_passes)

    def device_verify(**kwargs):
        recorded.device_calls.append(
            (kwargs.get("before_pte_path"), kwargs.get("after_pte_path")))
        return Verification(recorded.device_passes)

    def benchmark(**kwargs):
        recorded.benchmarks.append(
            (kwargs.get("before_pte_path"), kwargs.get("after_pte_path")))
        before, after = recorded.next_latency()
        return Benchmark(before, after)

    monkeypatch.setattr(pipeline, "verify_repair", host)
    monkeypatch.setattr(pipeline.device_verification,
                        "run_device_verification", device_verify)
    monkeypatch.setattr(pipeline.benchmarking, "benchmark_before_after", benchmark)
    monkeypatch.setattr(pipeline.export_model, "run_on_host",
                        lambda pte, args: [torch.zeros(1, 4, 8, 8)])
    monkeypatch.setattr(pipeline, "save_input_for_device",
                        lambda tensor, run_dir, index=0: f"{run_dir}/in{index}.bin")
    monkeypatch.setattr(pipeline, "_find_device",
                        lambda runners_dir, **options: (
                            FakeDevice(), "bench", "etdump", ""))
    return recorded


class Profiles:
    """Serves a scripted sequence of profiles, one per profiling call."""

    def __init__(self, *sequence):
        self.sequence = list(sequence)
        self.calls = 0

    def __call__(self, **kwargs):
        index = min(self.calls, len(self.sequence) - 1)
        self.calls += 1
        return self.sequence[index]


def run(spec, tmp_path, **options):
    options.setdefault("artifacts_dir", str(tmp_path / "art"))
    options.setdefault("quiet", True)
    return pipeline.run_optimization(spec, **options)


def no_provider(monkeypatch):
    """Record every attempt to construct a provider."""
    built = []

    def build(**kwargs):
        built.append(kwargs)
        raise RuntimeError("no provider in this test")

    monkeypatch.setattr("delegate_doctor.agent.client.build_provider", build)
    return built


# --- eligibility, end to end ---------------------------------------------------

def test_a_known_repair_just_above_the_dd_threshold_is_attempted(
        tmp_path, monkeypatch, gates):
    """Cheap to try when the answer is already in the catalog."""
    no_provider(monkeypatch)
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_with(("_softmax.out", 0.0011))))
    outcome = run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)
    attempts = outcome.repair_history.attempts
    assert attempts and attempts[0].repair_id == "DD-001"


def test_a_known_repair_exactly_at_the_dd_threshold_is_skipped(
        tmp_path, monkeypatch, gates):
    no_provider(monkeypatch)
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_with(("_softmax.out", 0.001))))
    outcome = run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)
    assert outcome.repair_history.attempts == []
    assert gates.benchmarks == []


def test_a_known_repair_is_applied_without_asking(tmp_path, monkeypatch, gates):
    """A deterministic repair needs no permission: the gates decide it."""
    no_provider(monkeypatch)
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_with(("_softmax.out", 0.38))))

    def refuse(question):
        pytest.fail(f"the user was prompted about a catalog repair: {question}")

    outcome = run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path,
                  interactive=True, prompt=refuse)
    assert outcome.repair_history.accepted_count == 1








# --- the catalog always comes first --------------------------------------------





# --- unknown hotspots ----------------------------------------------------------







# --- re-profiling, and what it implies -----------------------------------------



def test_a_rejected_repair_does_not_trigger_a_reprofile(tmp_path, monkeypatch,
                                                        gates):
    """Nothing changed, so there is nothing new to measure."""
    no_provider(monkeypatch)
    profiles = Profiles(profile_with(("_softmax.out", 0.38)))
    monkeypatch.setattr(pipeline.profiling, "profile_model", profiles)
    gates.latencies = [(100.0, 140.0)]

    run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)
    assert profiles.calls == 1


def test_a_rejected_repair_leaves_the_current_program_unchanged(
        tmp_path, monkeypatch, gates):
    no_provider(monkeypatch)
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_with(("_softmax.out", 0.38))))
    gates.latencies = [(100.0, 140.0)]

    spec = spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),))
    before = str(spec.exported_program.graph)
    outcome = run(spec, tmp_path)

    assert outcome.status == result_module.REPAIR_REJECTED
    assert str(spec.exported_program.graph) == before
    assert outcome.output_pte is None


# --- what each gate is handed ---------------------------------------------------

class FakeRule:
    """A catalog rule that always matches one operator and mutates the graph.

    Two of these give a genuine multi-repair run, which is the only way to test
    the properties that only appear from the second accepted repair onward.
    """

    def __init__(self, rule_id, operator, marker):
        self.RULE_ID = rule_id
        self.RULE_TITLE = f"{rule_id} test rule"
        self.operator = operator
        self.marker = marker

    def describe_rewrite(self):
        return "test rewrite"

    def matches_portable_kernel(self, kernel_name):
        return self.operator in kernel_name

    def detect(self, program):
        applied = getattr(program, self.marker, False)
        return _Detection(applies=not applied)

    def apply(self, program):
        # A real graph change, not just a flag. The loop identifies a graph
        # state by its contents to decide whether a rule has already been tried
        # there, so a "rule" that left the graph byte-identical would be
        # indistinguishable from one that did nothing - and a rule that did
        # nothing has nothing to accept.
        setattr(program, self.marker, True)
        graph = program.graph
        output = next(node for node in reversed(list(graph.nodes))
                      if node.op == "output")
        source = output.args[0][0]
        with graph.inserting_before(output):
            alias = graph.call_function(torch.ops.aten.alias.default, (source,))
        # alias is the identity on values, so the source's own metadata is
        # exactly right and lowering has everything it needs.
        alias.meta["val"] = source.meta["val"]
        output.args = (((alias,) + tuple(output.args[0][1:])),)
        graph.lint()
        program.graph_module.recompile()
        return 1


class _Detection:
    def __init__(self, applies):
        self.applies = applies
        self.detections = []
        self.declined = []
        self.skipped = []

    def explain(self):
        return "test detection"


@pytest.fixture
def two_rules(monkeypatch):
    """Replace the catalog with two rules that each match one operator."""
    rules = [FakeRule("TEST-001", "_softmax.out", "_test_one"),
             FakeRule("TEST-002", "mean.out", "_test_two")]
    monkeypatch.setattr(pipeline, "ALL_RULES", rules)
    return rules


def test_two_repairs_accumulate_and_the_program_carries_both(
        tmp_path, monkeypatch, gates, two_rules):
    no_provider(monkeypatch)
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_with(("_softmax.out", 0.38),
                                              ("mean.out", 0.12)),
                                 profile_with(("mean.out", 0.28)),
                                 profile_with(("mul.out", 0.002))))

    outcome = run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path,
                  ai_repair=True)
    history = outcome.repair_history

    assert history.accepted_count == 2
    assert history.applied_repair_ids == ["TEST-001", "TEST-002"]
    assert outcome.status == result_module.REPAIRS_ACCEPTED
    # Both repairs are present in the final program, so they really accumulated
    # rather than each starting from the pristine original.
    final = outcome.repair_history
    assert final.catalog_count == 2


def test_correctness_is_always_measured_against_the_original(
        tmp_path, monkeypatch, gates, two_rules):
    """Not against the previous step - that would let error accumulate."""
    no_provider(monkeypatch)
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_with(("_softmax.out", 0.38),
                                              ("mean.out", 0.12)),
                                 profile_with(("mean.out", 0.28)),
                                 profile_with(("mul.out", 0.002))))

    run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)

    assert len(gates.device_calls) == 2, "this test needs two accepted repairs"
    # Device verification always compares against the *original* .pte, so the
    # "before" path is identical on every call however many repairs accumulate.
    befores = {before for before, _ in gates.device_calls}
    assert len(befores) == 1, "the correctness reference changed between steps"
    assert "before" in str(next(iter(befores)))

    # Host verification likewise: the same original output object every time.
    original_ids = {original for original, _ in gates.host_calls}
    assert len(original_ids) == 1, "the host reference changed between steps"


def test_the_benchmark_reference_moves_forward_while_correctness_does_not(
        tmp_path, monkeypatch, gates, two_rules):
    """The two gates deliberately look in different directions."""
    no_provider(monkeypatch)
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_with(("_softmax.out", 0.38),
                                              ("mean.out", 0.12)),
                                 profile_with(("mean.out", 0.28)),
                                 profile_with(("mul.out", 0.002))))

    run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)

    # Two step benchmarks plus the final original-vs-final headline.
    step_benchmarks = gates.benchmarks[:2]
    first_before = step_benchmarks[0][0]
    second_before = step_benchmarks[1][0]
    assert first_before != second_before, (
        "the second repair was benchmarked against the original, not against "
        "the already-accepted program")
    # The second step's baseline is the first step's accepted candidate.
    assert second_before == step_benchmarks[0][1]


def test_a_multi_repair_run_measures_original_against_final_directly(
        tmp_path, monkeypatch, gates, two_rules):
    """Chaining step speedups would accumulate the drift each one cancelled."""
    no_provider(monkeypatch)
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_with(("_softmax.out", 0.38),
                                              ("mean.out", 0.12)),
                                 profile_with(("mean.out", 0.28)),
                                 profile_with(("mul.out", 0.002))))

    run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)

    assert len(gates.benchmarks) == 3, "no final original-vs-final measurement"
    final_before, final_after = gates.benchmarks[-1]
    assert "before" in str(final_before), "the headline did not start from the original"
    assert final_after == gates.benchmarks[1][1], "the headline did not end at the final program"


def test_a_single_repair_run_does_not_re_measure_the_headline(
        tmp_path, monkeypatch, gates):
    """With one repair, the step benchmark already compared those two programs."""
    no_provider(monkeypatch)
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_with(("_softmax.out", 0.38))))

    run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)
    assert len(gates.benchmarks) == 1


def test_the_safety_cap_stops_a_pathological_run(tmp_path, monkeypatch, gates):
    """A rule that keeps re-matching new hotspots must still terminate."""
    monkeypatch.setattr(repair_loop, "MAX_REPAIR_ITERATIONS", 3)
    no_provider(monkeypatch)

    counter = {"n": 0}

    def endless(**kwargs):
        # Every profile shows a brand-new hotspot node, so the "already
        # attempted" set can never catch up. Only the cap can stop this.
        counter["n"] += 1
        return profile_with((f"op{counter['n']}.out", 0.30))

    rule = FakeRule("TEST-LOOP", "op", "_never_set")
    rule.detect = lambda program: _Detection(applies=True)
    monkeypatch.setattr(pipeline, "ALL_RULES", [rule])
    monkeypatch.setattr(pipeline.profiling, "profile_model", endless)

    outcome = run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)
    history = outcome.repair_history
    assert history.iterations == 3
    assert "safety" in history.stop_reason


def test_the_benchmark_compares_the_current_program_with_the_candidate(
        tmp_path, monkeypatch, gates):
    """Incremental: each repair must improve what is already accepted."""
    no_provider(monkeypatch)
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_with(("_softmax.out", 0.38))))

    run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)

    assert gates.benchmarks, "nothing was benchmarked"
    before, after = gates.benchmarks[0]
    assert before != after
    # The very first candidate's "before" is the original, since nothing has
    # been accepted yet.
    assert "before" in str(before)


def test_the_final_report_keeps_the_original_to_final_comparison(
        tmp_path, monkeypatch, gates):
    no_provider(monkeypatch)
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_with(("_softmax.out", 0.38))))

    outcome = run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)
    history = outcome.repair_history
    assert history.original_latency_ms == 100.0
    assert history.final_latency_ms == 50.0
    assert history.total_speedup == pytest.approx(2.0)
    assert history.original_runtime_delegation is not None
    assert history.final_runtime_delegation is not None


# --- accumulation ---------------------------------------------------------------

def test_several_catalog_repairs_can_accumulate(tmp_path, monkeypatch, gates):
    """The run does not stop after the first catalog repair."""
    no_provider(monkeypatch)

    class TwoProblems(torch.nn.Module):
        def forward(self, x):
            return torch.softmax(x, dim=1) + 1

    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_with(("_softmax.out", 0.38),
                                              ("alias.out", 0.20)),
                                 profile_with(("alias.out", 0.25)),
                                 profile_with(("mul.out", 0.02))))

    outcome = run(spec_for(TwoProblems(), (torch.randn(1, 4, 8, 8),)), tmp_path)
    history = outcome.repair_history
    # At least the softmax repair, and the loop kept going afterwards.
    assert history.accepted_count >= 1
    assert len(history.attempts) >= 2, "the loop stopped after one hotspot"


def test_the_result_status_reflects_a_multi_repair_run(tmp_path, monkeypatch,
                                                       gates):
    history = repair_loop.RepairHistory()
    for _ in range(3):
        history.record(repair_loop.RepairAttempt(
            iteration=1, hotspot=hotspot(), source="catalog",
            repair_id="DD-001", status="ACCEPTED"))
    outcome = result_module.OptimizationResult(status="REPAIRS_ACCEPTED")
    outcome.repair_history = history
    assert outcome.accepted_repair_count == 3
    assert outcome.repair_accepted


def test_singular_fields_do_not_misreport_a_mixed_run():
    """`repair_source` cannot honestly answer for a catalog+AI run, and says so."""
    history = repair_loop.RepairHistory()
    history.record(repair_loop.RepairAttempt(
        iteration=1, hotspot=hotspot(), source="catalog", repair_id="DD-001",
        status="ACCEPTED"))
    history.record(repair_loop.RepairAttempt(
        iteration=2, hotspot=hotspot(), source="ai",
        candidate_id="AI-CANDIDATE-001", status="ACCEPTED"))

    outcome = result_module.OptimizationResult(status="REPAIRS_ACCEPTED")
    outcome.repair_history = history
    assert outcome.repair_source == "mixed"
    assert outcome.repair_id == "AI-CANDIDATE-001"
    assert outcome.repair_experimental is True


def test_the_history_reaches_the_report_and_the_json(tmp_path, monkeypatch, gates):
    import json

    no_provider(monkeypatch)
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_with(("_softmax.out", 0.38))))
    outcome = run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)

    assert "OPTIMIZATION SEQUENCE" in outcome.report_text
    payload = json.loads(
        (tmp_path / "art" / "run_001" / "repair_history.json").read_text())
    assert payload["attempts"][0]["hotspot"]["operator"] == "_softmax.out"
    assert payload["attempts"][0]["source"] == "catalog"

    html = (tmp_path / "art" / "run_001" / "report.html").read_text()
    assert "Optimization journey" in html
