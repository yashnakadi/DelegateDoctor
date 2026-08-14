"""AI repair as one bounded investigation of the whole model.

The architecture under test:

    known DD repairs first, always
        -> re-profile
    portable runtime still > 5% of the model?
        -> one consent prompt for the run
        -> one model-level provider request
        -> proposals become ordinary repair candidates
        -> the same gates a DD repair faces
        -> accepted candidate becomes current, re-profile, DDs again

What is deliberately *not* here any more: operator families, structural
patterns, per-site AI eligibility, coverage thresholds. Deciding which
combination of operators is worth repairing is the investigation, not a
precondition for starting it.

Fully offline: real `torch.export` graphs, a recording fake provider, mocked
device gates, no network.
"""

import json

import pytest
import torch

from delegate_doctor import (model_exploration, pipeline, repair_loop,
                             reporting, result as result_module)
from delegate_doctor.export_model import ModelSpec
from delegate_doctor.profiling import PortableKernel, ProfileResult

from tests.test_repair_loop import Profiles, gates, no_provider  # noqa: F401


# --- models ---------------------------------------------------------------------

class SoftmaxNet(torch.nn.Module):
    """DD-001 territory: a non-last-dim softmax."""

    def forward(self, x):
        return torch.softmax(x, dim=1)


class ScatteredNet(torch.nn.Module):
    """Several small fallbacks, no single one large."""

    def forward(self, x, mask):
        x = torch.fmod(x, 2.0)
        x = x + torch.erf(x)
        x = torch.where(mask > 0, x, x * 2.0)
        return x + mask.expand(x.shape)


def export(model, *args):
    return torch.export.export(model.eval(), tuple(args))


def spec_for(model, args, name="test model"):
    return ModelSpec(name=name, exported_program=export(model, *args),
                     example_args=args)


def scattered_spec():
    args = (torch.randn(2, 4, 8), torch.randn(2, 4, 8))
    return spec_for(ScatteredNet(), args, name="scattered")


def profile_of(*operators, delegated_share=None):
    """A profile with the named (operator, share) portable kernels."""
    portable = sum(share for _, share in operators)
    return ProfileResult(
        method_execute_ms=100.0,
        delegated_ms=(1.0 - portable) * 100,
        portable_ms=portable * 100,
        delegate_call_count=1,
        operator_call_count=len(operators),
        portable_kernels=[PortableKernel(
            name=f"native_call_{operator}", total_ms=share * 100,
            call_count=1, runtime_fraction=share, site_costs=(share * 100,))
            for operator, share in operators],
    )


def run(spec, tmp_path, **options):
    options.setdefault("artifacts_dir", str(tmp_path / "art"))
    options.setdefault("quiet", True)
    return pipeline.run_optimization(spec, **options)


class RecordingProvider:
    """Counts model-level requests and records exactly what was sent."""

    configuration = None

    def __init__(self, plan_text=None):
        self.requests = []
        self.plan_text = plan_text

    def complete(self, request):
        from delegate_doctor.agent.provider_response import (
            SUCCESS, ProviderCompletionResult)

        self.requests.append(request)
        return ProviderCompletionResult(
            SUCCESS, text=self.plan_text or "not a repair candidate")

    @property
    def sent(self) -> str:
        return "\n".join(request.user for request in self.requests)


@pytest.fixture
def provider(monkeypatch):
    recording = RecordingProvider()
    monkeypatch.setattr("delegate_doctor.agent.client.build_provider",
                        lambda **kwargs: recording)
    return recording


# --- eligibility is about the model ----------------------------------------------

@pytest.mark.parametrize("portable, eligible", [
    (0.040, False),     # 4% - not worth investigating
    (0.0500, False),    # exactly 5% - strictly greater-than
    (0.0501, True),
    (0.205, True),      # the Swin-T case
])
def test_eligibility_is_strictly_above_five_percent_portable(portable,
                                                             eligible):
    decision = model_exploration.assess(
        profile_of(("fmod.out", portable)))
    assert decision.eligible is eligible
    assert decision.portable_runtime_share == pytest.approx(portable)


def test_the_threshold_is_one_named_constant():
    assert model_exploration.MIN_AI_PORTABLE_RUNTIME_SHARE == 0.05


def test_several_small_operators_make_an_eligible_model():
    """3% + 2% + 2% = 7%. No individual operator clears the bar; the model does.

    This is the case a per-operator threshold could never admit, and it is
    exactly what AI exploration exists to look at.
    """
    decision = model_exploration.assess(
        profile_of(("fmod.out", 0.03), ("erf.out", 0.02), ("where.out", 0.02)))
    assert decision.eligible
    assert decision.portable_runtime_share == pytest.approx(0.07)


def test_an_ineligible_model_says_why():
    decision = model_exploration.assess(profile_of(("fmod.out", 0.02)))
    assert not decision.eligible
    assert "does not exceed 5%" in decision.reason


def test_no_profile_is_not_eligible():
    decision = model_exploration.assess(None)
    assert not decision.eligible
    assert "no device profile" in decision.reason


# --- the bounded model context -----------------------------------------------------

def build_context(spec, profile):
    from delegate_doctor.repairs import ALL_RULES

    return model_exploration.build_model_context(
        spec.exported_program, profile, None, ALL_RULES, "1.4.0")


def test_the_context_describes_the_whole_graph_not_one_neighbourhood():
    spec = scattered_spec()
    context, known = build_context(
        spec, profile_of(("fmod.out", 0.03), ("erf.out", 0.02)))

    targets = " ".join(entry.get("target", "")
                       for entry in context["graph"]["nodes"])
    # Several distinct portable regions, not one operator's window.
    for operator in ("fmod", "erf", "where", "expand"):
        assert operator in targets, operator
    assert len(known) >= 6


def test_the_context_marks_which_regions_are_portable():
    spec = scattered_spec()
    context, _ = build_context(
        spec, profile_of(("fmod.out", 0.03), ("erf.out", 0.02)))

    portable = [entry for entry in context["graph"]["nodes"]
                if entry.get("portable")]
    assert portable, "the request never says which regions XNNPACK declined"
    assert {"fmod", "erf"} <= {
        entry["target"].split("::")[-1] for entry in portable}


def test_the_context_carries_the_measurement():
    spec = scattered_spec()
    context, _ = build_context(
        spec, profile_of(("fmod.out", 0.03), ("erf.out", 0.02)))

    measurement = context["measurement"]
    assert measurement["portable_runtime"] == pytest.approx(0.05)
    assert measurement["runtime_delegation"] == pytest.approx(0.95)
    assert len(measurement["portable_operators"]) == 2


def test_the_context_names_the_repairs_delegate_doctor_already_knows():
    """So a request is not spent rediscovering DD-001."""
    spec = scattered_spec()
    context, _ = build_context(spec, profile_of(("fmod.out", 0.05)))

    known = {entry["id"] for entry in context["known_repairs"]}
    assert "DD-001" in known and "DD-002" in known
    assert all(entry["rewrite"] for entry in context["known_repairs"])


def test_the_context_asks_a_model_level_question():
    spec = scattered_spec()
    context, _ = build_context(spec, profile_of(("fmod.out", 0.05)))

    task = context["task"].lower()
    assert "analyze this exported model" in task
    assert "several operators" in task
    # Not a hotspot-shaped instruction.
    assert "fmod" not in task


def test_placeholders_are_described_by_role_never_by_parameter_name():
    """A placeholder's target is the parameter's name. It must not be sent."""
    class Parameterized(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.secret_layer_weight = torch.nn.Parameter(torch.randn(8))

        def forward(self, x):
            return torch.fmod(x * self.secret_layer_weight, 2.0)

    spec = spec_for(Parameterized(), (torch.randn(2, 4, 8),))
    context, _ = build_context(spec, profile_of(("fmod.out", 0.05)))

    text = json.dumps(context)
    assert "secret_layer_weight" not in text
    roles = {entry.get("role") for entry in context["graph"]["nodes"]}
    assert "placeholder" in roles


def test_the_context_carries_no_source_weights_or_tensor_values():
    spec = scattered_spec()
    context, _ = build_context(spec, profile_of(("fmod.out", 0.05)))
    text = json.dumps(context)

    for forbidden in ("class ScatteredNet", "def forward", "state_dict",
                      "Parameter containing", "0x", "torch.randn"):
        assert forbidden not in text, forbidden
    # No tensor data: shapes and dtypes only.
    for entry in context["graph"]["nodes"]:
        assert set(entry) <= {"id", "role", "target", "inputs", "users",
                              "literals", "shape", "dtype", "portable"}


def test_the_context_is_bounded(monkeypatch):
    monkeypatch.setattr(model_exploration, "MAX_DESCRIBED_NODES", 5)
    spec = scattered_spec()
    described = model_exploration.describe_graph(spec.exported_program,
                                                 limit=5)
    assert len(described) == 5


# --- one request per run, at the model level ----------------------------------------

def test_a_provider_is_asked_once_for_the_model(tmp_path, monkeypatch, gates,
                                                provider):
    """Not once per operator, per family, or per site."""
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_of(("fmod.out", 0.03),
                                            ("erf.out", 0.02),
                                            ("where.self_out", 0.02),
                                            ("expand_copy.out", 0.02))))

    run(scattered_spec(), tmp_path, ai_repair=True)

    # At most the existing per-exploration candidate bound - four portable
    # operators do not mean four investigations.
    assert len(provider.requests) <= 2, (
        f"{len(provider.requests)} requests for one model")


def test_the_request_covers_several_portable_regions(tmp_path, monkeypatch,
                                                     gates, provider):
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_of(("fmod.out", 0.03),
                                            ("erf.out", 0.02),
                                            ("where.self_out", 0.02))))

    run(scattered_spec(), tmp_path, ai_repair=True)

    sent = provider.sent
    for operator in ("fmod", "erf", "where", "expand"):
        assert operator in sent, operator


def test_the_flag_is_the_consent_and_nothing_prompts(tmp_path, monkeypatch,
                                                     gates, provider):
    """`--ai-repair` was typed. Asking again is a second confirmation."""
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_of(("fmod.out", 0.20))))
    asked = []

    outcome = run(scattered_spec(), tmp_path, ai_repair=True, interactive=True,
                  prompt=lambda question: asked.append(question) or "y")

    assert asked == [], f"the user was prompted anyway: {asked}"
    assert outcome.repair_history.ai_consent == repair_loop.AI_CONSENT_GRANTED
    assert provider.requests, "AI repair did not run despite --ai-repair"


def test_without_the_flag_no_ai_repair_happens(tmp_path, monkeypatch, gates,
                                               provider):
    """The default product: known repairs, then stop."""
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_of(("fmod.out", 0.20))))

    def refuse(question):
        pytest.fail(f"the user was prompted about AI repair: {question}")

    outcome = run(scattered_spec(), tmp_path, interactive=True, prompt=refuse)

    assert provider.requests == []
    assert outcome.repair_history.ai_consent == \
        repair_loop.AI_CONSENT_NOT_ENABLED


def test_a_model_below_the_bar_is_never_offered(tmp_path, monkeypatch, gates):
    built = no_provider(monkeypatch)
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_of(("fmod.out", 0.03))))

    def refuse(question):
        pytest.fail("the user was asked about a model below the threshold")

    outcome = run(scattered_spec(), tmp_path, interactive=True, prompt=refuse)
    assert built == []
    assert outcome.status == result_module.NO_REPAIR_AVAILABLE


def test_an_eligible_model_with_no_provider_says_unavailable(tmp_path,
                                                             monkeypatch,
                                                             gates, capsys):
    no_provider(monkeypatch)
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_of(("fmod.out", 0.20))))

    outcome = run(scattered_spec(), tmp_path, quiet=False, ai_repair=True)

    printed = capsys.readouterr().out
    assert "AI exploration          unavailable" in printed
    assert "no configured AI provider" in printed
    assert outcome.repair_history.ai_consent == repair_loop.AI_CONSENT_UNAVAILABLE


# --- DD priority ---------------------------------------------------------------------

def test_ai_is_never_reached_while_a_known_repair_applies(tmp_path, monkeypatch,
                                                          gates, provider):
    profiles = Profiles(
        profile_of(("_softmax.out", 0.38)),
        profile_of())                          # repaired: nothing portable left
    monkeypatch.setattr(pipeline.profiling, "profile_model", profiles)

    outcome = run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path,
                  ai_repair=True)

    assert provider.requests == [], "AI ran while a catalog repair applied"
    accepted = outcome.repair_history.accepted
    assert accepted and accepted[0].source == repair_loop.SOURCE_CATALOG


def test_an_accepted_ai_repair_returns_to_dd_priority(tmp_path, monkeypatch,
                                                      gates, provider):
    """A re-profile after AI may expose a known repair. It goes next."""
    import copy

    def accept_once(self, iteration, history):
        if getattr(self, "_did", False):
            return None
        self._did = True
        attempt = repair_loop.RepairAttempt(
            iteration=iteration, hotspot=None,
            source=repair_loop.SOURCE_AI, candidate_id="AI-CANDIDATE-001")
        self._evaluate(copy.deepcopy(self.current_program), attempt)
        history.record(attempt)
        return True if attempt.accepted else None

    monkeypatch.setattr(pipeline._RepairMachinery, "_explore_model", accept_once)
    monkeypatch.setattr(pipeline.profiling, "profile_model", Profiles(
        profile_of(("fmod.out", 0.20)),           # nothing known: AI first
        profile_of(("_softmax.out", 0.38)),       # now DD-001 applies
        profile_of()))

    outcome = run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path,
                  ai_repair=True)

    sources = [attempt.source for attempt in outcome.repair_history.attempts]
    assert sources[0] == repair_loop.SOURCE_AI
    assert repair_loop.SOURCE_CATALOG in sources[1:], (
        "a known repair exposed by the re-profile did not take priority")


# --- proposals are ordinary candidates -------------------------------------------------

def valid_plan(anchor, before):
    return json.dumps({
        "summary": "insert a clone",
        "anchor": anchor,
        "operations": [
            {"type": "insert_aten_call", "id": "new_1",
             "target": "aten.clone.default",
             "args": [{"node": anchor}], "before": before},
        ]})


def plan_provider(monkeypatch, spec, operator="fmod"):
    """A provider whose reply is a valid plan anchored on a real node."""
    from delegate_doctor import operator_correlation

    nodes = list(spec.exported_program.graph.nodes)
    target = operator_correlation.candidate_nodes(
        spec.exported_program, operator)[0]
    index = nodes.index(target)
    recording = RecordingProvider(valid_plan(f"node_{index}",
                                             f"node_{index + 1}"))
    monkeypatch.setattr("delegate_doctor.agent.client.build_provider",
                        lambda **kwargs: recording)
    return recording


def test_a_runnable_proposal_meets_the_same_gates_as_a_dd(tmp_path, monkeypatch,
                                                          gates):
    spec = scattered_spec()
    plan_provider(monkeypatch, spec)
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_of(("fmod.out", 0.20)), profile_of()))

    outcome = run(spec, tmp_path, ai_repair=True)

    attempt = outcome.repair_history.attempts[0]
    assert attempt.source == repair_loop.SOURCE_AI
    assert attempt.host_verification_passed is True
    assert attempt.device_verification_passed is True
    assert attempt.before_latency_ms and attempt.after_latency_ms
    assert gates.host_calls and gates.device_calls and gates.benchmarks


def test_correct_and_faster_is_accepted(tmp_path, monkeypatch, gates):
    spec = scattered_spec()
    plan_provider(monkeypatch, spec)
    gates.latencies = [(100.0, 50.0)]
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_of(("fmod.out", 0.20)), profile_of()))

    outcome = run(spec, tmp_path, ai_repair=True)
    assert outcome.repair_history.accepted
    assert outcome.status in (result_module.REPAIR_ACCEPTED,
                              result_module.REPAIRS_ACCEPTED)


def test_correct_but_slower_is_rejected(tmp_path, monkeypatch, gates):
    spec = scattered_spec()
    plan_provider(monkeypatch, spec)
    gates.latencies = [(100.0, 140.0)]
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_of(("fmod.out", 0.20))))

    outcome = run(spec, tmp_path, ai_repair=True)
    assert outcome.repair_history.attempts[0].status == repair_loop.REJECTED
    assert outcome.status == result_module.REPAIR_REJECTED


def test_incorrect_but_faster_is_rejected(tmp_path, monkeypatch, gates):
    spec = scattered_spec()
    plan_provider(monkeypatch, spec)
    gates.host_passes = False
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_of(("fmod.out", 0.20))))

    outcome = run(spec, tmp_path, ai_repair=True)
    attempt = outcome.repair_history.attempts[0]
    assert attempt.status == repair_loop.REJECTED
    assert attempt.host_verification_passed is False


def test_a_structurally_invalid_proposal_is_rejected_without_execution(
        tmp_path, monkeypatch, gates, provider):
    """The provider's reply is prose. Nothing is applied, nothing is measured."""
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_of(("fmod.out", 0.20))))

    outcome = run(scattered_spec(), tmp_path, ai_repair=True)

    statuses = {attempt.status for attempt in outcome.repair_history.attempts}
    assert statuses == {repair_loop.NO_CANDIDATE}
    assert gates.host_calls == []
    assert gates.device_calls == []
    assert gates.benchmarks == []


def test_an_ai_attempt_is_not_tied_to_one_profiled_hotspot():
    """A model-level proposal is about the graph, not about an operator row."""
    attempt = repair_loop.RepairAttempt(
        iteration=1, hotspot=None, source=repair_loop.SOURCE_AI,
        candidate_id="AI-CANDIDATE-001")
    assert attempt.subject == "model-level exploration"
    assert attempt.to_dict()["hotspot"] is None


# --- terminal and summary --------------------------------------------------------------

def test_normal_terminal_output_has_no_family_or_correlation_vocabulary(
        tmp_path, monkeypatch, gates, provider, capsys):
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_of(("fmod.out", 0.20))))

    run(scattered_spec(), tmp_path, quiet=False, ai_repair=True)

    printed = capsys.readouterr().out
    for banned in ("Family runtime", "Correlated runtime", "Compatible sites",
                   "Runtime covered", "structural pattern", "operator family",
                   "sites 11 of 11", "equivalent sites"):
        assert banned not in printed, banned


def test_the_terminal_shows_the_exploration_and_its_candidates(
        tmp_path, monkeypatch, gates, provider, capsys):
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_of(("fmod.out", 0.20))))

    run(scattered_spec(), tmp_path, quiet=False, ai_repair=True)

    printed = capsys.readouterr().out
    assert "Exploring model for experimental AI repairs..." in printed
    assert "AI-CANDIDATE-001" in printed
    assert "Result                 NO_CANDIDATE" in printed


def test_the_enabled_screen_is_model_level(tmp_path, monkeypatch, gates,
                                           provider, capsys):
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_of(("fmod.out", 0.205))))
    shown = []

    run(scattered_spec(), tmp_path, quiet=False, ai_repair=True)
    shown.append(capsys.readouterr().out)

    screen = shown[0]
    assert "Runtime delegation      79.5%" in screen
    assert "Portable runtime        20.5%" in screen
    assert "No known DelegateDoctor repairs remain." in screen
    assert "Experimental AI repair enabled (--ai-repair)." in screen
    # The disclosure survives: consent moved to the flag, the obligation to
    # say what leaves the machine did not.
    assert "DelegateDoctor will NOT send:" in screen
    for never in ("model source", "weights", "tensor values",
                  "representative inputs", "checkpoints", "API keys"):
        assert never in screen, never
    # And it is a statement, not a question.
    assert "[y/N]" not in screen


def test_the_summary_reports_ai_exploration_without_stale_dd_state(
        tmp_path, monkeypatch, gates, provider):
    """A run that explored AI must not advertise a DD candidate."""
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_of(("fmod.out", 0.20))))

    outcome = run(scattered_spec(), tmp_path, ai_repair=True)

    summary = reporting.format_summary(outcome)
    assert "AI exploration" in summary
    assert "DD-001" not in summary
    assert "DD-002" not in summary


@pytest.mark.parametrize("scenario", ["not_attempted", "no_candidate"])
def test_the_final_result_distinguishes_what_happened(scenario, tmp_path,
                                                      monkeypatch, gates,
                                                      provider):
    if scenario == "not_attempted":
        monkeypatch.setattr(pipeline.profiling, "profile_model",
                            Profiles(profile_of(("fmod.out", 0.03))))
        outcome = run(scattered_spec(), tmp_path, ai_repair=True)
        assert outcome.repair_history.ai_consent == \
            repair_loop.AI_CONSENT_NOT_NEEDED
        assert outcome.repair_history.attempts == []
    else:
        monkeypatch.setattr(pipeline.profiling, "profile_model",
                            Profiles(profile_of(("fmod.out", 0.20))))
        outcome = run(scattered_spec(), tmp_path, ai_repair=True)
        assert outcome.repair_history.attempts
        assert all(attempt.status == repair_loop.NO_CANDIDATE
                   for attempt in outcome.repair_history.attempts)


def test_the_history_and_the_report_tell_the_same_story(tmp_path, monkeypatch,
                                                        gates, provider):
    from delegate_doctor import html_report

    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_of(("fmod.out", 0.20))))

    outcome = run(scattered_spec(), tmp_path, ai_repair=True)

    assert "OPTIMIZATION SEQUENCE" in outcome.report_text
    assert "model-level exploration" in outcome.report_text

    payload = json.loads(json.dumps(outcome.repair_history.to_dict()))
    assert payload["attempts"][0]["subject"] == "model-level exploration"
    assert payload["attempts"][0]["source"] == "ai"
    for stale in ("family_runtime_share", "correlated_runtime_share",
                  "covered_runtime_share", "pattern_id"):
        assert stale not in payload["attempts"][0], stale

    html = html_report._journey(outcome)
    assert "AI-CANDIDATE-001" in html
    for banned in ("Family runtime", "Correlated runtime", "Runtime covered"):
        assert banned not in html, banned


# --- the old architecture is gone ---------------------------------------------------

def test_the_operator_family_module_no_longer_exists():
    with pytest.raises(ImportError):
        __import__("delegate_doctor.operator_family")


def test_no_production_module_references_the_old_ai_architecture():
    import inspect
    from pathlib import Path

    package = Path(inspect.getfile(model_exploration)).parent
    retired = ("operator_family", "hotspot_pattern",
               "MIN_AI_OPERATOR_RUNTIME_SHARE", "MIN_AI_REPAIR_COVERAGE",
               "MIN_AI_HOTSPOT_RUNTIME_SHARE", "covered_runtime_share",
               "correlated_runtime_share", "family_runtime_share",
               "structural_signature", "build_family_context")
    for path in package.rglob("*.py"):
        text = path.read_text()
        for token in retired:
            assert token not in text, f"{path.name} still references {token}"


def test_correlation_survives_as_an_internal_utility():
    """Still needed to map a runtime hotspot to a node for catalog matching."""
    from delegate_doctor import operator_correlation

    spec = scattered_spec()
    resolution = operator_correlation.resolve_hotspot(
        spec.exported_program, "expand_copy.out")
    assert resolution.resolved
    assert "expand" in resolution.node_id


# --- --ai-repair is an opt-in, and only for repair -----------------------------------

def unrepairable_heavy_profile():
    """The reported shape: 60% portable runtime and no known repair.

    This used to name `avg_pool2d.out`, taken from the Inception V3 run where
    that kernel was 60.7% of runtime and nothing in the catalog matched it.
    DD-003 now matches exactly that kernel, so keeping the old name would have
    quietly turned these into "a rule was attempted" tests instead of the "no
    rule matched" tests they are. The operators below are the ones ScatteredNet
    actually contains, and no catalog rule claims either.
    """
    return profile_of(("fmod.out", 0.607), ("erf.out", 0.001))


@pytest.mark.parametrize("portable", [0.06, 0.20, 0.607])
def test_no_flag_means_no_provider_call_at_any_portable_share(
        portable, tmp_path, monkeypatch, gates, provider):
    """Cases 2/3/20: 60% portable is still not a reason to call a provider."""
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_of(("fmod.out", portable))))

    outcome = run(scattered_spec(), tmp_path, interactive=True,
                  prompt=lambda question: pytest.fail("prompted"))

    assert provider.requests == [], "AI repair ran without --ai-repair"
    assert outcome.status == result_module.NO_REPAIR_AVAILABLE


def test_the_inception_shape_makes_no_provider_call_by_default(
        tmp_path, monkeypatch, gates, provider):
    """Case 20: the exact reported profile, default flags."""
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(unrepairable_heavy_profile()))

    run(scattered_spec(), tmp_path, interactive=True,
        prompt=lambda question: pytest.fail("prompted"))
    assert len(provider.requests) == 0


def test_the_inception_shape_makes_exactly_one_call_with_the_flag(
        tmp_path, monkeypatch, gates, provider):
    """Case 21: one model-level exploration, not one per hotspot."""
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(unrepairable_heavy_profile()))

    run(scattered_spec(), tmp_path, ai_repair=True)
    # The per-exploration candidate bound still applies; what must not happen
    # is a request per portable operator.
    assert 1 <= len(provider.requests) <= 2


def test_a_known_repair_runs_without_the_flag_and_calls_no_provider(
        tmp_path, monkeypatch, gates, provider):
    """Case 1: the default product is DD repair, and it works."""
    profiles = Profiles(profile_of(("_softmax.out", 0.38)), profile_of())
    monkeypatch.setattr(pipeline.profiling, "profile_model", profiles)

    outcome = run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)

    assert provider.requests == []
    accepted = outcome.repair_history.accepted
    assert accepted and accepted[0].source == repair_loop.SOURCE_CATALOG


def test_the_flag_still_puts_known_repairs_first(tmp_path, monkeypatch, gates,
                                                 provider):
    """Case 5: --ai-repair does not reorder anything."""
    profiles = Profiles(profile_of(("_softmax.out", 0.38)), profile_of())
    monkeypatch.setattr(pipeline.profiling, "profile_model", profiles)

    outcome = run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path,
                  ai_repair=True)

    assert provider.requests == [], "AI ran while a known repair applied"
    assert outcome.repair_history.accepted[0].source == repair_loop.SOURCE_CATALOG


def test_the_flag_works_non_interactively(tmp_path, monkeypatch, gates,
                                          provider):
    """Case 12: no second prompt, so the flag is usable in CI."""
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_of(("fmod.out", 0.20))))

    run(scattered_spec(), tmp_path, interactive=False, ai_repair=True,
        prompt=lambda question: pytest.fail("prompted non-interactively"))
    assert provider.requests


def test_no_flag_non_interactively_never_runs_ai(tmp_path, monkeypatch, gates,
                                                 provider):
    """Case 13."""
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(profile_of(("fmod.out", 0.20))))

    run(scattered_spec(), tmp_path, interactive=False)
    assert provider.requests == []


def test_the_python_api_does_not_enable_ai_repair_by_default():
    """Case 14."""
    import inspect

    signature = inspect.signature(pipeline.run_optimization)
    assert signature.parameters["ai_repair"].default is False


def test_the_default_summary_has_no_ai_clutter(tmp_path, monkeypatch, gates,
                                               provider):
    """Case 17: an opt-in feature must not report itself as a skipped step."""
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(unrepairable_heavy_profile()))

    outcome = run(scattered_spec(), tmp_path)
    summary = reporting.format_summary(outcome)

    assert "AI exploration" not in summary
    assert "Candidates tested" not in summary
    assert "NO REPAIR AVAILABLE" in summary


def test_the_default_sequence_states_what_happened(tmp_path, monkeypatch,
                                                   gates, provider):
    """Cases 10/22: never claim portable runtime was below the AI threshold."""
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(unrepairable_heavy_profile()))

    outcome = run(scattered_spec(), tmp_path)
    text = outcome.report_text

    assert "No known DelegateDoctor repair matched." in text
    assert "Experimental AI repair was not enabled." in text
    for stale in ("portable runtime below the AI threshold",
                  "portable runtime does not exceed",
                  "No portable hotspot exceeded"):
        assert stale not in text, stale


def test_no_stale_threshold_claim_survives_anywhere():
    """Case 22, at the source."""
    import inspect
    from pathlib import Path

    package = Path(inspect.getfile(pipeline)).parent
    for path in package.rglob("*.py"):
        text = path.read_text()
        assert "portable runtime below the AI threshold" not in text, path.name


def test_the_result_wording_names_known_repairs(tmp_path, monkeypatch, gates,
                                                provider):
    """Case 11."""
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(unrepairable_heavy_profile()))

    outcome = run(scattered_spec(), tmp_path)
    assert "No known DelegateDoctor repair matches" in outcome.summary


def test_the_report_section_omits_ai_when_it_was_not_enabled(tmp_path,
                                                             monkeypatch,
                                                             gates, provider):
    """Case 9/17: no "AI exploration Not requested" on a default run."""
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(unrepairable_heavy_profile()))

    outcome = run(scattered_spec(), tmp_path)
    assert "AI exploration" not in outcome.report_text


def test_the_report_section_records_ai_when_it_was_enabled(tmp_path,
                                                           monkeypatch,
                                                           gates, provider):
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        Profiles(unrepairable_heavy_profile()))

    outcome = run(scattered_spec(), tmp_path, ai_repair=True)
    assert "AI exploration" in outcome.report_text
