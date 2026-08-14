"""Phase 6: what an AI may propose, and everything it may not do.

The point of nearly every test here is a refusal. The schema and the applier
exist so that "the model returned something dangerous" is a validation error
rather than an incident, and so the pristine baseline survives whatever is
attempted against it.
"""

import copy
import json

import pytest
import torch

from delegate_doctor.agent import graph_context, repair_applier, repair_schema
from delegate_doctor.agent.repair_applier import CandidateApplicationError
from delegate_doctor.agent.repair_schema import CandidateValidationError


class SoftmaxNet(torch.nn.Module):
    """A graph with a non-last-dimension softmax to point candidates at."""

    def forward(self, x):
        return torch.softmax(x * 2.0, dim=1)


@pytest.fixture
def program():
    return torch.export.export(SoftmaxNet().eval(), (torch.randn(1, 4, 8, 8),))


@pytest.fixture
def node_ids(program):
    return [f"node_{index}" for index, _ in enumerate(program.graph.nodes)]


def candidate(**overrides):
    plan = {
        "summary": "insert a reshape before the softmax",
        "anchor": "node_2",
        "operations": [
            {"type": "insert_aten_call", "id": "new_1",
             "target": "aten.reshape.default",
             "args": [{"node": "node_1"}, [1, 4, 64]],
             "before": "node_2"},
        ],
    }
    plan.update(overrides)
    return plan


# --- the graph neighbourhood --------------------------------------------------

def test_the_neighbourhood_is_bounded(program):
    view = graph_context.build_neighbourhood(program, "_softmax", radius=2)
    assert len(view.nodes) <= graph_context.MAX_NEIGHBOURHOOD_NODES
    assert len(view.nodes) <= 5


def test_the_hotspot_is_identified_and_marked(program):
    view = graph_context.build_neighbourhood(program, "_softmax")
    assert view.hotspot_identifier
    marked = [node for node in view.nodes if node.is_hotspot]
    assert len(marked) == 1
    assert "softmax" in marked[0].target.lower()


def test_shapes_are_sent_but_never_values(program):
    view = graph_context.build_neighbourhood(program, "_softmax")
    payload = json.dumps(view.to_dict())
    assert '"shape"' in payload
    # A real tensor's values would appear as long float lists.
    assert "0.0," not in payload
    assert "tensor(" not in payload


def test_no_memory_addresses_or_object_reprs_are_included(program):
    payload = json.dumps(
        graph_context.build_neighbourhood(program, "_softmax").to_dict())
    assert "0x" not in payload
    assert " object at " not in payload


def test_a_large_literal_is_summarized_not_transmitted():
    assert graph_context._safe_literal(list(range(500))).startswith("<sequence")
    assert graph_context._safe_literal("x" * 500).startswith("<str len=")


def test_an_unmatched_hotspot_yields_an_empty_neighbourhood(program):
    view = graph_context.build_neighbourhood(program, "definitely_not_here")
    assert view.nodes == []
    assert view.hotspot_identifier == ""


def test_the_repair_context_carries_measurement_but_no_inputs(program):
    class Kernel:
        operator_name = "_softmax.out"
        total_ms = 48.2
        runtime_fraction = 0.634

    class Profile:
        portable_kernels = [Kernel()]
        runtime_delegation_fraction = 0.343

    class Delegation:
        operator_delegation_fraction = 0.968
        portable_op_total = 1
        total_ops = 41
        delegate_blob_count = 2

    view = graph_context.build_neighbourhood(program, "_softmax")
    context = graph_context.build_repair_context(view, Profile(), Delegation(),
                                                 "1.4.0")
    payload = json.dumps(context)

    assert context["measurement"]["hotspot_ms"] == 48.2
    assert context["measurement"]["runtime_delegation"] == 0.343
    assert "graph" in context
    for forbidden in ("weight", "state_dict", "input0.bin", "randn"):
        assert forbidden not in payload


# --- the candidate schema -----------------------------------------------------

def test_a_valid_candidate_parses(node_ids):
    plan = repair_schema.parse_candidate(candidate(), node_ids)
    assert plan.anchor == "node_2"
    assert plan.new_node_ids == ["new_1"]


def test_fenced_json_is_tolerated(node_ids):
    text = "```json\n" + json.dumps(candidate()) + "\n```"
    assert repair_schema.parse_candidate_text(text, node_ids).anchor == "node_2"


@pytest.mark.parametrize("text", [
    "import torch; graph.erase_node(n)",
    "Here is the rewrite you asked for.",
    "",
    "{broken json",
])
def test_python_and_prose_replies_are_rejected(text, node_ids):
    with pytest.raises(CandidateValidationError):
        repair_schema.parse_candidate_text(text, node_ids)


def test_an_unknown_operation_is_rejected(node_ids):
    with pytest.raises(CandidateValidationError) as caught:
        repair_schema.parse_candidate(
            candidate(operations=[{"type": "run_shell", "command": "rm -rf /"}]),
            node_ids)
    assert "unknown type" in str(caught.value)


def test_an_unknown_aten_target_is_rejected_before_any_mutation(node_ids):
    with pytest.raises(CandidateValidationError) as caught:
        repair_schema.parse_candidate(candidate(operations=[
            {"type": "insert_aten_call", "id": "new_1",
             "target": "aten.system.default", "args": [], "before": "node_2"}]),
            node_ids)
    assert "not on DelegateDoctor's allowlist" in str(caught.value)


@pytest.mark.parametrize("target", [
    "os.system", "subprocess.run", "torch.load", "aten.custom_op.default",
    "builtins.eval", "aten.reshape.evil",
])
def test_operators_outside_the_allowlist_are_refused(target, node_ids):
    with pytest.raises(CandidateValidationError):
        repair_schema.parse_candidate(candidate(operations=[
            {"type": "insert_aten_call", "id": "new_1", "target": target,
             "args": [], "before": "node_2"}]), node_ids)


def test_string_arguments_are_refused(node_ids):
    """No ATen call on the allowlist needs one, and code would hide there."""
    with pytest.raises(CandidateValidationError) as caught:
        repair_schema.parse_candidate(candidate(operations=[
            {"type": "insert_aten_call", "id": "new_1",
             "target": "aten.reshape.default",
             "args": [{"node": "node_1"}, "__import__('os').system('id')"],
             "before": "node_2"}]), node_ids)
    assert "may not be a string" in str(caught.value)


def test_an_unknown_node_reference_is_rejected(node_ids):
    with pytest.raises(CandidateValidationError) as caught:
        repair_schema.parse_candidate(candidate(operations=[
            {"type": "insert_aten_call", "id": "new_1",
             "target": "aten.reshape.default",
             "args": [{"node": "node_9999"}], "before": "node_2"}]), node_ids)
    assert "unknown node" in str(caught.value)


def test_a_duplicate_generated_id_is_rejected(node_ids):
    operation = {"type": "insert_aten_call", "id": "new_1",
                 "target": "aten.relu.default", "args": [{"node": "node_1"}],
                 "before": "node_2"}
    with pytest.raises(CandidateValidationError) as caught:
        repair_schema.parse_candidate(
            candidate(operations=[operation, dict(operation)]), node_ids)
    assert "duplicate node id" in str(caught.value)


def test_a_non_finite_literal_is_rejected(node_ids):
    with pytest.raises(CandidateValidationError):
        repair_schema.parse_candidate(candidate(operations=[
            {"type": "insert_aten_call", "id": "new_1",
             "target": "aten.mul.Tensor",
             "args": [{"node": "node_1"}, float("nan")], "before": "node_2"}]),
            node_ids)


def test_an_oversized_candidate_is_rejected(node_ids):
    operations = [
        {"type": "insert_aten_call", "id": f"new_{index}",
         "target": "aten.relu.default", "args": [{"node": "node_1"}],
         "before": "node_2"}
        for index in range(repair_schema.MAX_AI_REPAIR_OPERATIONS + 3)
    ]
    with pytest.raises(CandidateValidationError) as caught:
        repair_schema.parse_candidate(candidate(operations=operations), node_ids)
    assert "above the" in str(caught.value)


def test_too_many_new_nodes_is_rejected(node_ids):
    operations = [
        {"type": "insert_aten_call", "id": f"new_{index}",
         "target": "aten.relu.default", "args": [{"node": "node_1"}],
         "before": "node_2"}
        for index in range(repair_schema.MAX_NEW_NODES + 1)
    ]
    with pytest.raises(CandidateValidationError) as caught:
        repair_schema.parse_candidate(candidate(operations=operations), node_ids)
    assert "creates" in str(caught.value)


def test_an_anchor_outside_the_neighbourhood_is_rejected(node_ids):
    with pytest.raises(CandidateValidationError) as caught:
        repair_schema.parse_candidate(candidate(anchor="node_9999"), node_ids)
    assert "not a node in the supplied neighbourhood" in str(caught.value)


def test_unknown_top_level_fields_are_rejected(node_ids):
    with pytest.raises(CandidateValidationError):
        repair_schema.parse_candidate(candidate(execute="rm -rf /"), node_ids)


def test_an_empty_operation_list_is_rejected(node_ids):
    with pytest.raises(CandidateValidationError):
        repair_schema.parse_candidate(candidate(operations=[]), node_ids)


# --- applying a candidate -----------------------------------------------------

def relu_candidate():
    """A real, harmless rewrite: insert a clone and route uses through it."""
    return {
        "summary": "insert a clone",
        "anchor": "node_1",
        "operations": [
            {"type": "insert_aten_call", "id": "new_1",
             "target": "aten.clone.default",
             "args": [{"node": "node_1"}], "before": "node_2"},
        ],
    }


def test_a_candidate_is_applied_to_a_copy_not_the_baseline(program, node_ids):
    before_nodes = len(list(program.graph.nodes))
    plan = repair_schema.parse_candidate(relu_candidate(), node_ids)

    rewritten = repair_applier.apply_candidate(program, plan)

    assert rewritten is not program
    assert len(list(program.graph.nodes)) == before_nodes, "baseline was mutated"
    assert len(list(rewritten.graph.nodes)) == before_nodes + 1


def test_each_candidate_starts_from_the_pristine_baseline(program, node_ids):
    """A failed attempt must not contaminate the next one."""
    before_nodes = len(list(program.graph.nodes))
    plan = repair_schema.parse_candidate(relu_candidate(), node_ids)

    first = repair_applier.apply_candidate(program, plan)
    second = repair_applier.apply_candidate(program, plan)

    assert len(list(first.graph.nodes)) == before_nodes + 1
    assert len(list(second.graph.nodes)) == before_nodes + 1
    assert repair_applier.baseline_is_unchanged(program, before_nodes)


def test_the_baseline_still_computes_the_same_thing_afterwards(program, node_ids):
    inputs = (torch.randn(1, 4, 8, 8),)
    with torch.no_grad():
        before = program.module()(*inputs)

    plan = repair_schema.parse_candidate(relu_candidate(), node_ids)
    repair_applier.apply_candidate(program, plan)

    with torch.no_grad():
        after = program.module()(*inputs)
    assert torch.equal(before, after)


def test_erasing_a_graph_input_is_refused(program, node_ids):
    placeholders = [f"node_{index}" for index, node
                    in enumerate(program.graph.nodes) if node.op == "placeholder"]
    plan = repair_schema.parse_candidate(
        candidate(operations=[{"type": "erase_node", "node": placeholders[0]}]),
        node_ids)
    with pytest.raises(CandidateApplicationError) as caught:
        repair_applier.apply_candidate(program, plan)
    assert "graph input" in str(caught.value)


def test_erasing_the_output_is_refused(program, node_ids):
    outputs = [f"node_{index}" for index, node
               in enumerate(program.graph.nodes) if node.op == "output"]
    plan = repair_schema.parse_candidate(
        candidate(operations=[{"type": "erase_node", "node": outputs[0]}]),
        node_ids)
    with pytest.raises(CandidateApplicationError) as caught:
        repair_applier.apply_candidate(program, plan)
    assert "graph output" in str(caught.value)


def test_erasing_a_node_that_is_still_used_is_refused(program, node_ids):
    used = None
    for index, node in enumerate(program.graph.nodes):
        if node.op == "call_function" and list(node.users):
            used = f"node_{index}"
            break
    plan = repair_schema.parse_candidate(
        candidate(operations=[{"type": "erase_node", "node": used}]), node_ids)
    with pytest.raises(CandidateApplicationError) as caught:
        repair_applier.apply_candidate(program, plan)
    assert "still has users" in str(caught.value)


def test_an_allowlisted_target_resolves_to_a_real_overload():
    resolved = repair_applier._resolve_target("aten.reshape.default")
    assert resolved is torch.ops.aten.reshape.default


def test_resolution_refuses_anything_off_the_allowlist():
    for name in ("aten.system.default", "os.system", "aten.reshape.evil"):
        with pytest.raises(CandidateValidationError):
            repair_applier._resolve_target(name)


def test_the_applier_never_executes_provider_text():
    """The structural guarantee for Phase 6."""
    import ast

    for module in (repair_applier, repair_schema, graph_context):
        tree = ast.parse(open(module.__file__).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in ("eval", "exec", "compile"), \
                    f"{module.__name__} can execute text"


def test_the_applier_uses_no_dynamic_attribute_lookup_on_provider_strings():
    """`getattr(torch.ops.aten, name)` is guarded by allowlist membership."""
    source = open(repair_applier.__file__).read()
    assert "if name not in ALLOWED_ATEN_TARGETS" in source
    assert "importlib" not in source
    assert "__import__" not in source


# --- the catalog keeps priority ------------------------------------------------

def test_a_successful_candidate_is_not_added_to_the_catalog():
    """An AI repair is accepted for one run, never promoted to a rule."""
    from delegate_doctor.repairs import ALL_RULES

    assert [rule.RULE_ID for rule in ALL_RULES] == ["DD-001", "DD-002", "DD-003"]


def test_no_repair_module_is_generated_by_the_agent():
    from pathlib import Path

    repairs_dir = Path(repair_applier.__file__).parent.parent / "repairs"
    modules = sorted(path.name for path in repairs_dir.glob("*.py"))
    assert modules == ["__init__.py", "dd001_softmax.py", "dd002_noop_alias.py",
                       "dd003_avgpool_pad.py"]


def test_the_agent_cannot_write_into_the_repair_catalog():
    for module in (repair_applier, repair_schema):
        source = open(module.__file__).read()
        assert "ALL_RULES" not in source
        assert "repairs/" not in source


# --- Phase 6D/6E: when exploration may run, and what it may send -------------

from delegate_doctor import pipeline, result as result_module          # noqa: E402
from delegate_doctor.agent import repair_explorer                       # noqa: E402
from delegate_doctor.export_model import ModelSpec                      # noqa: E402
from tests.fake_provider import FakeProvider, RefusingProvider          # noqa: E402


class PlainNet(torch.nn.Module):
    """Fully delegable: nothing for anyone to repair."""

    def forward(self, x):
        return x + x


class FmodNet(torch.nn.Module):
    """A portable fallback that no catalog rule matches."""

    def forward(self, x):
        return torch.fmod(x, 2.0)


class FakeDevice:
    serial = "test-target"
    is_emulator = False

    def short_description(self):
        return "TestTarget · arm64-v8a"

    def describe(self):
        return "Arm64 Android device - TestTarget"


def fake_profile(kernels=("native_call_fmod.out",), portable_ms=8.2):
    from delegate_doctor.profiling import PortableKernel, ProfileResult

    return ProfileResult(
        method_execute_ms=20.0, delegated_ms=20.0 - portable_ms,
        portable_ms=portable_ms, delegate_call_count=1,
        operator_call_count=len(kernels),
        portable_kernels=[
            PortableKernel(name=name, total_ms=portable_ms, call_count=1,
                           runtime_fraction=portable_ms / 20.0)
            for name in kernels
        ],
    )


def spec_for(model, args):
    return ModelSpec(name="test model",
                     exported_program=torch.export.export(model.eval(), args),
                     example_args=args)


@pytest.fixture
def profiled_target(monkeypatch):
    """A device that profiles, so the AI fallback branch is reachable."""
    monkeypatch.setattr(pipeline, "_find_device",
                        lambda runners_dir, **options: (
                            FakeDevice(), "bench", "etdump", ""))
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        lambda **kwargs: fake_profile())


def run(spec, tmp_path, **options):
    options.setdefault("artifacts_dir", str(tmp_path / "art"))
    options.setdefault("quiet", True)
    return pipeline.run_optimization(spec, **options)



@pytest.fixture
def mocked_gates(monkeypatch):
    """Stand in for the device gates, recording that the *existing* ones ran.

    Deliberately patched at the pipeline's own call sites: the point is that an
    AI candidate goes through `verify_repair`, `run_device_verification`,
    `benchmark_before_after` and `decide_repair` - the same functions DD-001
    uses - not through anything written for AI.
    """
    calls = {"host": 0, "device": 0, "benchmark": 0, "decision": 0}

    class Metrics:
        max_absolute_error = 1.0e-08
        mean_absolute_error = 1.0e-09
        mean_squared_error = 1.0e-17
        max_relative_error = 1.0e-07

    class Verification:
        """`passed` is repair fidelity; backend fidelity is separate."""

        def __init__(self, passed=True, backend_fidelity="OK"):
            self.passed = passed
            self.repaired_vs_original = Metrics()
            self.repaired_vs_eager = Metrics()
            self.original_device_vs_host = Metrics()
            self.repaired_device_vs_host = Metrics()
            self.argmax_agreement = None
            self.backend_fidelity = backend_fidelity
            self.backend_fidelity_reason = ""
            self.failure_reasons = ["mocked gate failure"] if not passed else []
            self.error = ""

        @property
        def backend_fidelity_acceptable(self):
            return self.backend_fidelity != "FAIL"

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
        def __init__(self, before=100.0, after=50.0):
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

    state = {"host_passes": True, "device_passes": True,
             "before": 100.0, "after": 50.0}

    def host(**kwargs):
        calls["host"] += 1
        return Verification(state["host_passes"])

    def device(**kwargs):
        calls["device"] += 1
        return Verification(state["device_passes"])

    def benchmark(**kwargs):
        calls["benchmark"] += 1
        return Benchmark(state["before"], state["after"])

    real_decide = pipeline.decide_repair

    def decide(**kwargs):
        calls["decision"] += 1
        return real_decide(**kwargs)

    monkeypatch.setattr(pipeline, "verify_repair", host)
    monkeypatch.setattr(pipeline.device_verification,
                        "run_device_verification", device)
    monkeypatch.setattr(pipeline.benchmarking, "benchmark_before_after", benchmark)
    monkeypatch.setattr(pipeline, "decide_repair", decide)
    monkeypatch.setattr(pipeline.export_model, "run_on_host",
                        lambda pte, args: [torch.zeros(1, 4, 8, 8)])
    monkeypatch.setattr(pipeline, "save_input_for_device",
                        lambda tensor, run_dir, index=0: f"{run_dir}/in{index}.bin")

    calls["state"] = state
    return calls


def watch_provider(monkeypatch, provider):
    """Record whether the pipeline ever built a provider."""
    built = []

    def build(**kwargs):
        built.append(kwargs)
        return provider

    monkeypatch.setattr("delegate_doctor.agent.client.build_provider", build)
    return built


# --- exploration must not run --------------------------------------------------

def test_no_exploration_when_a_catalog_rule_matches(tmp_path, monkeypatch,
                                                    profiled_target, mocked_gates):
    """DD-001 wins outright: no request is made while it still applies.

    The repaired model is fully delegated afterwards, so nothing is left to
    investigate and AI is never reached at all.
    """
    built = watch_provider(monkeypatch, RefusingProvider())
    profiles = iter([
        fake_profile(("native_call__softmax.out",)),
        fake_profile(kernels=(), portable_ms=0.0),      # repaired: all delegated
    ])
    last = {}

    def profile(**kwargs):
        last["value"] = next(profiles, last.get("value"))
        return last["value"]

    monkeypatch.setattr(pipeline.profiling, "profile_model", profile)

    outcome = run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path,
                  ai_repair=True)
    assert built == [], "AI was consulted while a catalog repair applied"
    assert outcome.repair_source in (None, "catalog")


def test_no_exploration_when_fully_delegated(tmp_path, monkeypatch,
                                             profiled_target):
    built = watch_provider(monkeypatch, RefusingProvider())
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        lambda **kwargs: fake_profile(kernels=(), portable_ms=0.0))
    outcome = run(spec_for(PlainNet(), (torch.randn(1, 4, 8, 8),)), tmp_path,
                  ai_repair=True)
    assert built == []
    assert outcome.status == result_module.NO_REPAIR_REQUIRED


def test_no_exploration_without_opt_in(tmp_path, monkeypatch, profiled_target):
    built = watch_provider(monkeypatch, RefusingProvider())
    outcome = run(spec_for(FmodNet(), (torch.randn(1, 4, 8, 8),)), tmp_path,
                  interactive=False)
    assert built == []
    assert outcome.status == result_module.NO_REPAIR_AVAILABLE


def test_a_deterministic_run_never_builds_a_provider(tmp_path, monkeypatch,
                                                     profiled_target):
    """No --allow-ai and no prompt available: AI is simply never reached."""
    built = watch_provider(monkeypatch, RefusingProvider())
    outcome = run(spec_for(FmodNet(), (torch.randn(1, 4, 8, 8),)), tmp_path,
                  interactive=False)
    assert built == []
    assert outcome.status == result_module.NO_REPAIR_AVAILABLE


def test_no_exploration_without_a_device(tmp_path, monkeypatch):
    """Exploration needs a measured hotspot, which needs profiling."""
    built = watch_provider(monkeypatch, RefusingProvider())
    monkeypatch.setattr(pipeline, "_find_device",
                        lambda runners_dir, **options: (None, None, None, "none"))
    run(spec_for(FmodNet(), (torch.randn(1, 4, 8, 8),)), tmp_path, ai_repair=True)
    assert built == []


def test_a_missing_credential_degrades_gracefully(tmp_path, monkeypatch,
                                                  profiled_target):
    from delegate_doctor.agent.client import AINotConfigured

    def build(**kwargs):
        raise AINotConfigured("no key")

    monkeypatch.setattr("delegate_doctor.agent.client.build_provider", build)
    outcome = run(spec_for(FmodNet(), (torch.randn(1, 4, 8, 8),)), tmp_path,
                  ai_repair=True)
    assert outcome.status == result_module.NO_REPAIR_AVAILABLE
    assert outcome.ai_repair_requested


# --- what exploration sends ------------------------------------------------------

def test_the_repair_request_carries_graph_and_measurement(tmp_path, monkeypatch,
                                                          profiled_target):
    provider = FakeProvider("not json", "not json")
    watch_provider(monkeypatch, provider)
    run(spec_for(FmodNet(), (torch.randn(1, 4, 8, 8),)), tmp_path, ai_repair=True)

    assert provider.call_count >= 1
    # A model-level request: the graph, what it cost, and what DelegateDoctor
    # already knows - not one operator's neighbourhood.
    provider.assert_sent("portable", "runtime_delegation", "shape", "node_",
                         "known_repairs", "portable_operators")


def test_the_repair_request_never_carries_source_or_tensors(
        tmp_path, monkeypatch, profiled_target):
    """Phase 6 builds its own payload; it never reuses Phase 5's source."""
    provider = FakeProvider("not json", "not json")
    watch_provider(monkeypatch, provider)
    run(spec_for(FmodNet(), (torch.randn(1, 4, 8, 8),)), tmp_path, ai_repair=True)

    provider.assert_never_sent(
        "class FmodNet", "def forward", "import torch",   # model source
        "input0.bin", "state_dict", "weights",            # tensors and weights
        "DD_TEST_SUPER_SECRET",                            # credentials
    )


def test_the_repair_request_is_bounded(tmp_path, monkeypatch, profiled_target):
    provider = FakeProvider("not json", "not json")
    watch_provider(monkeypatch, provider)
    run(spec_for(FmodNet(), (torch.randn(1, 4, 8, 8),)), tmp_path, ai_repair=True)
    assert len(provider.requests[0].user) < 40_000


# --- candidate bounds ------------------------------------------------------------

def test_exploration_stops_at_the_candidate_limit(tmp_path, monkeypatch,
                                                  profiled_target):
    provider = FakeProvider(*(["not json"] * 10))
    watch_provider(monkeypatch, provider)
    outcome = run(spec_for(FmodNet(), (torch.randn(1, 4, 8, 8),)), tmp_path,
                  ai_repair=True)

    assert provider.call_count == repair_explorer.MAX_AI_REPAIR_CANDIDATES == 2
    assert outcome.ai_candidate_count == 2
    assert outcome.status == result_module.NO_REPAIR_AVAILABLE


def test_a_provider_cannot_raise_its_own_limit(tmp_path, monkeypatch,
                                               profiled_target):
    sneaky = json.dumps({"summary": "s", "anchor": "node_0",
                         "operations": [], "max_candidates": 99})
    provider = FakeProvider(sneaky, sneaky, sneaky, sneaky)
    watch_provider(monkeypatch, provider)
    run(spec_for(FmodNet(), (torch.randn(1, 4, 8, 8),)), tmp_path, ai_repair=True)
    assert provider.call_count == 2


def test_invalid_candidates_are_recorded_as_attempts(tmp_path, monkeypatch,
                                                     profiled_target):
    provider = FakeProvider("not json", "still not json")
    watch_provider(monkeypatch, provider)
    outcome = run(spec_for(FmodNet(), (torch.randn(1, 4, 8, 8),)), tmp_path,
                  ai_repair=True)

    assert len(outcome.ai_attempt_summaries) == 2
    assert all(entry["outcome"] == "invalid"
               for entry in outcome.ai_attempt_summaries)


def test_a_malformed_response_never_crashes_the_run(tmp_path, monkeypatch,
                                                    profiled_target):
    provider = FakeProvider("<html>gateway error</html>", "{}")
    watch_provider(monkeypatch, provider)
    outcome = run(spec_for(FmodNet(), (torch.randn(1, 4, 8, 8),)), tmp_path,
                  ai_repair=True)
    assert outcome.status == result_module.NO_REPAIR_AVAILABLE
    assert outcome.exit_code == 0


def test_the_pristine_baseline_survives_exploration(tmp_path, monkeypatch,
                                                    profiled_target):
    spec = spec_for(FmodNet(), (torch.randn(1, 4, 8, 8),))
    before = len(list(spec.exported_program.graph.nodes))

    provider = FakeProvider("not json", "not json")
    watch_provider(monkeypatch, provider)
    run(spec, tmp_path, ai_repair=True)

    assert len(list(spec.exported_program.graph.nodes)) == before


# --- the explorer itself ---------------------------------------------------------

def test_the_explorer_feeds_failures_back(tmp_path):
    provider = FakeProvider("not json", "still bad")
    result = repair_explorer.explore(
        provider=provider, baseline_program=None, context={"x": 1},
        known_nodes=["node_0"], lower=lambda program: None,
        announce=lambda *a: None)

    assert result.candidate_count == 2
    assert not result.found_runnable
    assert "rejected" in provider.requests[1].user.lower() or \
        "not usable" in provider.requests[1].user.lower()


def test_a_candidate_that_cannot_lower_is_retried(program):
    valid = json.dumps(relu_candidate())
    provider = FakeProvider(valid, valid)

    def always_fails(rewritten):
        raise RuntimeError("edge dialect rejected the graph")

    result = repair_explorer.explore(
        provider=provider, baseline_program=program, context={},
        known_nodes=[f"node_{i}" for i, _ in enumerate(program.graph.nodes)],
        lower=always_fails, announce=lambda *a: None)

    assert result.candidate_count == 2
    assert all(a.outcome == "not-lowerable" for a in result.attempts)
    assert not result.found_runnable


def test_a_runnable_candidate_is_returned_for_the_gates(program):
    provider = FakeProvider(json.dumps(relu_candidate()))
    result = repair_explorer.explore(
        provider=provider, baseline_program=program, context={},
        known_nodes=[f"node_{i}" for i, _ in enumerate(program.graph.nodes)],
        lower=lambda rewritten: None, announce=lambda *a: None)

    assert result.found_runnable
    assert result.plan.candidate_id == "AI-CANDIDATE-001"
    assert provider.call_count == 1


def test_each_candidate_is_applied_to_the_pristine_baseline(program):
    """Candidate 2 must not build on candidate 1."""
    applied = []
    valid = json.dumps(relu_candidate())
    provider = FakeProvider(valid, valid)

    original_nodes = len(list(program.graph.nodes))

    def record_then_fail(rewritten):
        applied.append(len(list(rewritten.graph.nodes)))
        raise RuntimeError("nope")

    repair_explorer.explore(
        provider=provider, baseline_program=program, context={},
        known_nodes=[f"node_{i}" for i, _ in enumerate(program.graph.nodes)],
        lower=record_then_fail, announce=lambda *a: None)

    # Both candidates saw a graph exactly one node larger than the pristine one.
    assert applied == [original_nodes + 1, original_nodes + 1]
    assert len(list(program.graph.nodes)) == original_nodes


# --- metadata --------------------------------------------------------------------

def test_result_metadata_distinguishes_ai_from_catalog():
    outcome = result_module.OptimizationResult(status="REPAIR_ACCEPTED")
    assert outcome.repair_source is None
    assert outcome.repair_experimental is False

    payload = outcome.to_dict()
    for field in ("repair_source", "repair_id", "repair_experimental",
                  "ai_provider", "ai_model", "ai_candidate_count"):
        assert field in payload


def test_no_raw_prompt_or_response_is_stored(tmp_path, monkeypatch,
                                             profiled_target, mocked_gates):
    plan_text = json.dumps(relu_candidate())
    provider = FakeProvider(plan_text, plan_text)
    watch_provider(monkeypatch, provider)

    outcome = run(spec_for(FmodNet(), (torch.randn(1, 4, 8, 8),)), tmp_path,
                  ai_repair=True)

    payload = json.dumps(outcome.to_dict()) + outcome.report_text
    assert "BEGIN UNTRUSTED" not in payload
    assert plan_text not in payload
    assert "You are DelegateDoctor's repair-exploration assistant" not in payload


# --- a runnable AI candidate faces the ordinary gates -------------------------

def ai_candidate_run(tmp_path, monkeypatch, gates, **state):
    """Drive a run where the AI proposes a valid, lowerable candidate."""
    gates["state"].update(state)
    plan = json.dumps({
        "summary": "insert a clone",
        "anchor": "node_1",
        "operations": [
            {"type": "insert_aten_call", "id": "new_1",
             "target": "aten.clone.default",
             "args": [{"node": "node_1"}], "before": "node_2"},
        ],
    })
    provider = FakeProvider(plan, plan)
    watch_provider(monkeypatch, provider)
    outcome = run(spec_for(FmodNet(), (torch.randn(1, 4, 8, 8),)), tmp_path,
                  ai_repair=True)
    return outcome, provider


def test_a_runnable_candidate_uses_the_existing_gates(tmp_path, monkeypatch,
                                                      profiled_target,
                                                      mocked_gates):
    outcome, _ = ai_candidate_run(tmp_path, monkeypatch, mocked_gates)
    assert mocked_gates["host"] == 1
    assert mocked_gates["device"] == 1
    assert mocked_gates["benchmark"] == 1
    assert mocked_gates["decision"] == 1
    assert outcome.repair_source == "ai"
    assert outcome.repair_experimental is True


def test_correct_and_faster_is_accepted(tmp_path, monkeypatch, profiled_target,
                                        mocked_gates):
    outcome, _ = ai_candidate_run(tmp_path, monkeypatch, mocked_gates,
                                  before=100.0, after=50.0)
    assert outcome.status == result_module.REPAIR_ACCEPTED
    assert outcome.repair_id == "AI-CANDIDATE-001"


def test_correct_but_slower_is_rejected(tmp_path, monkeypatch, profiled_target,
                                        mocked_gates):
    outcome, _ = ai_candidate_run(tmp_path, monkeypatch, mocked_gates,
                                  before=50.0, after=100.0)
    assert outcome.status == result_module.REPAIR_REJECTED


def test_faster_but_host_incorrect_is_rejected(tmp_path, monkeypatch,
                                               profiled_target, mocked_gates):
    outcome, _ = ai_candidate_run(tmp_path, monkeypatch, mocked_gates,
                                  host_passes=False, before=100.0, after=10.0)
    assert outcome.status == result_module.REPAIR_REJECTED


def test_faster_but_device_incorrect_is_rejected(tmp_path, monkeypatch,
                                                 profiled_target, mocked_gates):
    outcome, _ = ai_candidate_run(tmp_path, monkeypatch, mocked_gates,
                                  device_passes=False, before=100.0, after=10.0)
    assert outcome.status == result_module.REPAIR_REJECTED


def test_no_second_candidate_after_a_genuine_gate_rejection(
        tmp_path, monkeypatch, profiled_target, mocked_gates):
    """A correctness failure is an answer, not a prompt for another guess."""
    outcome, provider = ai_candidate_run(tmp_path, monkeypatch, mocked_gates,
                                         host_passes=False)
    assert provider.call_count == 1
    assert outcome.status == result_module.REPAIR_REJECTED


def test_an_accepted_ai_candidate_is_not_added_to_the_catalog(
        tmp_path, monkeypatch, profiled_target, mocked_gates):
    from delegate_doctor.repairs import ALL_RULES

    before = [rule.RULE_ID for rule in ALL_RULES]
    ai_candidate_run(tmp_path, monkeypatch, mocked_gates)
    assert [rule.RULE_ID for rule in ALL_RULES] == before == ["DD-001", "DD-002",
                                                             "DD-003"]


def test_the_summary_labels_the_repair_as_experimental(
        tmp_path, monkeypatch, profiled_target, mocked_gates):
    from delegate_doctor import reporting

    outcome, _ = ai_candidate_run(tmp_path, monkeypatch, mocked_gates)
    summary = reporting.format_summary(outcome)
    assert "Accepted repairs        1" in summary
    assert "AI                      1" in summary
    assert "AI-CANDIDATE-001" in summary
    assert "Experimental            Yes" in summary


# --- the decision screen reaches the pipeline ---------------------------------

def test_the_pipeline_builds_the_opportunity_summary(tmp_path, profiled_target):
    """Every profiled run carries the summary, whether or not AI is involved."""
    outcome = run(spec_for(FmodNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)
    summary = outcome.opportunity
    assert summary is not None
    assert summary.has_measurement
    assert summary.top_hotspot.operator == "fmod.out"
    # 8.2 of 20.0 ms: the ceiling is 1 / (1 - 0.41).
    assert summary.theoretical_upper_bound_speedup == pytest.approx(
        1 / (1 - 0.41), abs=1e-6)


def test_the_summary_matches_the_profile_it_came_from(tmp_path, profiled_target):
    outcome = run(spec_for(FmodNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)
    profile = outcome.before_profile
    assert outcome.opportunity.portable_runtime_ms == profile.portable_ms
    assert (outcome.opportunity.runtime_delegation
            == profile.runtime_delegation_fraction)


def test_a_catalog_match_is_named_in_the_summary(tmp_path, monkeypatch,
                                                 profiled_target, mocked_gates):
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        lambda **kwargs: fake_profile(("native_call__softmax.out",)))
    outcome = run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)
    assert "DD-001" in outcome.opportunity.catalog_match


def test_the_report_text_carries_the_opportunity(tmp_path, profiled_target):
    outcome = run(spec_for(FmodNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)
    assert "REPAIR OPPORTUNITY" in outcome.report_text
    assert "Theoretical upper bound" in outcome.report_text




def test_a_run_that_never_asked_says_not_requested(tmp_path, profiled_target):
    outcome = run(spec_for(FmodNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)
    assert outcome.opportunity.ai_status == "Not requested"




def test_a_small_operator_family_is_never_offered(tmp_path, monkeypatch,
                                                  profiled_target):
    """2% of runtime does not justify a provider request under the family bar."""
    built = watch_provider(monkeypatch, RefusingProvider())
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        lambda **kwargs: fake_profile(portable_ms=0.4))

    def refuse(question):
        pytest.fail("the user was asked about a family below the threshold")

    outcome = run(spec_for(FmodNet(), (torch.randn(1, 4, 8, 8),)), tmp_path,
                  interactive=True, prompt=refuse)

    assert built == []
    assert outcome.repair_history.attempts == []


def test_an_unavailable_provider_is_recorded_and_not_fatal(tmp_path, monkeypatch,
                                                           profiled_target):
    """No key configured: the analysis still completes and says why."""
    from delegate_doctor.agent.client import AINotConfigured

    def refuse(**kwargs):
        raise AINotConfigured("AI NOT CONFIGURED")

    monkeypatch.setattr("delegate_doctor.agent.client.build_provider", refuse)
    outcome = run(spec_for(FmodNet(), (torch.randn(1, 4, 8, 8),)), tmp_path,
                  ai_repair=True)
    assert outcome.status == result_module.NO_REPAIR_AVAILABLE
    assert outcome.opportunity.ai_status == "Unavailable"


def test_a_non_interactive_run_without_the_flag_resolves_no_provider(
        tmp_path, monkeypatch, profiled_target):
    """Consent is impossible, so nothing - not even a key lookup - happens."""
    built = watch_provider(monkeypatch, RefusingProvider())
    outcome = run(spec_for(FmodNet(), (torch.randn(1, 4, 8, 8),)), tmp_path,
                  interactive=False, ai_repair=False)
    assert built == []
    assert outcome.opportunity.ai_status == "Not requested"
    assert not outcome.ai_repair_requested


def test_ai_repair_is_off_unless_the_flag_is_given(tmp_path, monkeypatch,
                                                   profiled_target):
    """The default product never reaches a provider."""
    built = watch_provider(monkeypatch, RefusingProvider())

    def refuse(question):
        pytest.fail(f"the user was prompted about AI repair: {question}")

    outcome = run(spec_for(FmodNet(), (torch.randn(1, 4, 8, 8),)), tmp_path,
                  interactive=True, prompt=refuse)

    assert built == []
    assert not outcome.ai_repair_requested
    assert outcome.repair_history.ai_consent == "not enabled"


def test_the_flag_replaces_the_consent_prompt(tmp_path, monkeypatch,
                                              profiled_target, capsys):
    """`--ai-repair` authorizes; the notice states what is sent."""
    watch_provider(monkeypatch, RefusingProvider())

    def refuse(question):
        pytest.fail(f"a redundant prompt appeared: {question}")

    run(spec_for(FmodNet(), (torch.randn(1, 4, 8, 8),)), tmp_path,
        quiet=False, ai_repair=True, interactive=True, prompt=refuse)

    printed = capsys.readouterr().out
    assert "Experimental AI repair enabled (--ai-repair)." in printed
    assert "DelegateDoctor will NOT send:" in printed
    assert "[y/N]" not in printed
