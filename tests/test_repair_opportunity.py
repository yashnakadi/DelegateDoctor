"""The decision screen: enough context to answer "is this worth an API call?".

Every number here is derived once, in `repair_opportunity`, and rendered by
three surfaces. These tests hold two lines: the arithmetic must be right and
guarded, and the three surfaces must not disagree.

Fully offline - no model, no device, no provider.
"""

import pytest

from delegate_doctor import html_report, repair_opportunity
from delegate_doctor.agent import consent
from delegate_doctor.result import OptimizationResult


# --- stand-ins for the measured objects -------------------------------------

class FakeKernel:
    def __init__(self, name="_softmax.out", total_ms=48.2, fraction=0.634):
        self.name = f"native_call_{name}"
        self.operator_name = name
        self.total_ms = total_ms
        self.call_count = 1
        self.runtime_fraction = fraction


class FakeProfile:
    def __init__(self, kernels=None, runtime_delegation=0.343, portable_ms=50.0,
                 method_execute_ms=76.0):
        self.method_execute_ms = method_execute_ms
        self.portable_ms = portable_ms
        self.delegated_ms = 26.0
        self.delegate_call_count = 1
        self.operator_call_count = 3
        self.portable_kernels = (kernels if kernels is not None
                                 else [FakeKernel()])
        self.accounting_warning = ""
        self._runtime = runtime_delegation

    @property
    def runtime_delegation_fraction(self):
        return self._runtime


class FakeDelegation:
    def __init__(self, total=100, portable=1):
        self.total_ops = total
        self.portable_op_total = portable
        self.delegated_op_total = total - portable
        self.delegate_blob_count = 1

    @property
    def operator_delegation_fraction(self):
        return self.delegated_op_total / self.total_ops


class FakeDefinition:
    label = "Anthropic"


class FakeConfiguration:
    definition = FakeDefinition()
    model = "claude-sonnet-4-5"
    is_local = False


def summary(**overrides):
    built = repair_opportunity.build_summary(
        profile=overrides.pop("profile", FakeProfile()),
        delegation=overrides.pop("delegation", FakeDelegation()),
        target=overrides.pop("target", "TestTarget (arm64-v8a)"),
        catalog_match=overrides.pop("catalog_match", "None"),
        configuration=overrides.pop("configuration", None),
    )
    for key, value in overrides.items():
        setattr(built, key, value)
    return built


# --- the arithmetic ----------------------------------------------------------

def test_the_upper_bound_is_amdahls_ceiling_for_the_top_hotspot():
    """63.4% of runtime removed entirely leaves 36.6%: 1 / (1 - f)."""
    built = summary()
    assert built.theoretical_upper_bound_speedup == pytest.approx(
        1 / (1 - 0.634))


def test_the_upper_bound_is_omitted_when_the_hotspot_is_all_of_runtime():
    """f = 1 would divide by zero. No number is better than an infinity."""
    built = summary(profile=FakeProfile(kernels=[FakeKernel(fraction=1.0)]))
    assert built.theoretical_upper_bound_speedup is None


def test_the_upper_bound_is_omitted_for_a_nonsensical_fraction():
    built = summary(profile=FakeProfile(kernels=[FakeKernel(fraction=1.4)]))
    assert built.theoretical_upper_bound_speedup is None
    built = summary(profile=FakeProfile(kernels=[FakeKernel(fraction=0.0)]))
    assert built.theoretical_upper_bound_speedup is None


def test_there_is_no_upper_bound_without_a_hotspot():
    built = summary(profile=FakeProfile(kernels=[]))
    assert built.top_hotspot is None
    assert built.theoretical_upper_bound_speedup is None


def test_portable_runtime_is_the_complement_of_runtime_delegation():
    built = summary(profile=FakeProfile(runtime_delegation=0.343))
    assert built.portable_runtime_fraction == pytest.approx(0.657)


def test_a_hotspots_share_of_fallback_is_guarded_against_no_fallback():
    """A profile with kernels but no portable time must not divide by zero."""
    built = summary(profile=FakeProfile(kernels=[FakeKernel(total_ms=0.0)],
                                        portable_ms=0.0))
    assert built.top_hotspot.portable_fraction == 0.0


def test_a_hotspots_share_of_fallback_is_relative_to_portable_time():
    profile = FakeProfile(kernels=[FakeKernel(total_ms=40.0),
                                   FakeKernel("mul.out", 10.0, 0.13)],
                          portable_ms=50.0)
    built = summary(profile=profile)
    assert built.top_hotspot.portable_fraction == pytest.approx(0.8)


def test_hotspots_are_ranked_by_measured_cost_not_input_order():
    profile = FakeProfile(kernels=[FakeKernel("mul.out", 2.0, 0.02),
                                   FakeKernel("_softmax.out", 40.0, 0.5),
                                   FakeKernel("add.out", 9.0, 0.1)])
    built = summary(profile=profile)
    assert built.top_hotspot.operator == "_softmax.out"
    assert [other.operator for other in built.other_hotspots] == ["add.out",
                                                                  "mul.out"]


def test_only_a_few_secondary_hotspots_are_listed():
    kernels = [FakeKernel(f"op{index}.out", 10.0 - index, 0.1)
               for index in range(10)]
    built = summary(profile=FakeProfile(kernels=kernels))
    assert len(built.other_hotspots) == repair_opportunity.OTHER_HOTSPOT_LIMIT


def test_nothing_is_invented_when_nothing_was_measured():
    built = repair_opportunity.build_summary()
    assert built.runtime_delegation is None
    assert built.operator_delegation is None
    assert built.portable_runtime_fraction is None
    assert built.theoretical_upper_bound_speedup is None
    assert not built.has_measurement


def test_milliseconds_come_from_the_profiler_not_from_a_percentage():
    """Mixing the benchmark p50 with an ETDump share would be two rulers."""
    built = summary(profile=FakeProfile(portable_ms=50.0,
                                        method_execute_ms=76.0))
    assert built.portable_runtime_ms == 50.0
    assert built.measured_latency_ms == 76.0
    assert built.top_hotspot.runtime_ms == 48.2


# --- what the screen says ----------------------------------------------------

def test_the_decision_screen_gives_the_numbers_needed_to_decide():
    text = repair_opportunity.format_decision_screen(summary())
    for expected in ("Runtime delegation", "34.3%",      # measured delegation
                     "Portable execution", "65.7%",      # what is left over
                     "_softmax.out", "48.200 ms", "63.4%"):
        assert expected in text, expected


def test_the_screen_never_promises_a_speedup():
    """A ceiling is not a prediction, and must never be phrased as one."""
    text = repair_opportunity.format_decision_screen(summary())
    lowered = text.lower()
    for forbidden in ("expected speedup", "predicted speedup", "will be faster",
                      "estimated speedup", "you will get"):
        assert forbidden not in lowered, forbidden


def test_the_bound_lives_in_the_report_rather_than_the_consent_screen():
    """The screen asks "may I send this?"; the bound is reading material."""
    built = summary(profile=FakeProfile(kernels=[FakeKernel(fraction=0.03)]))
    screen = repair_opportunity.format_decision_screen(built)
    report = repair_opportunity.format_report_section(built)
    assert "Theoretical upper bound" not in screen
    assert "Theoretical upper bound 1.03x" in report


def test_the_screen_carries_no_explanatory_paragraph():
    """Concise: measurements and a bound, not an essay about Amdahl."""
    text = repair_opportunity.format_decision_screen(summary())
    for forbidden in ("ceiling if this operator", "dropped to zero",
                      "nothing else changed", "maximum possible end-to-end",
                      "not a prediction"):
        assert forbidden not in text, forbidden


def test_the_screen_still_states_exactly_what_is_and_is_not_sent():
    """More context must not have displaced the privacy disclosure."""
    text = repair_opportunity.format_decision_screen(summary())
    assert "It will send:" in text and "It will NOT send:" in text
    for kept in ("your model source", "model weights", "tensor values",
                 "representative inputs", "checkpoints", "API keys"):
        assert kept in text, kept


def test_the_screen_names_the_provider_it_would_send_to():
    text = repair_opportunity.format_decision_screen(
        summary(configuration=FakeConfiguration()))
    assert "Anthropic" in text and "claude-sonnet-4-5" in text


def test_a_local_provider_is_described_as_local():
    class Local(FakeConfiguration):
        is_local = True

    text = repair_opportunity.format_decision_screen(
        summary(configuration=Local()))
    assert "nothing leaves this machine" in text
    assert "your local provider" in text


def test_the_screen_omits_what_was_never_measured():
    """A missing number is left out, never printed as a measured zero."""
    text = repair_opportunity.format_decision_screen(
        repair_opportunity.build_summary())
    assert "0.0%" not in text
    assert "0.000 ms" not in text


def test_the_screen_carries_no_emoji():
    text = repair_opportunity.format_decision_screen(summary())
    assert all(character.isascii() or character in "·—" for character in text)


# --- one derivation, three surfaces ------------------------------------------

def test_the_terminal_and_report_text_agree_on_every_number():
    built = summary()
    screen = repair_opportunity.format_decision_screen(built)
    report = repair_opportunity.format_report_section(built)
    for shared in ("34.3%", "65.7%", "_softmax.out", "48.200 ms", "63.4%"):
        assert shared in screen and shared in report, shared


def test_the_html_report_renders_from_the_same_object():
    outcome = OptimizationResult(status="NO_REPAIR_AVAILABLE",
                                 model_name="Fake")
    outcome.opportunity = summary()
    html = html_report._repair_opportunity(outcome)
    for shared in ("34.3%", "65.7%", "_softmax.out", "48.200 ms", "63.4%",
                   "2.73x"):
        assert shared in html, shared


def test_the_html_section_is_absent_without_a_measurement():
    outcome = OptimizationResult(status="ANALYSIS_COMPLETE")
    assert html_report._repair_opportunity(outcome) == ""
    outcome.opportunity = repair_opportunity.build_summary()
    assert html_report._repair_opportunity(outcome) == ""


def test_the_html_section_escapes_an_operator_name():
    outcome = OptimizationResult(status="NO_REPAIR_AVAILABLE")
    outcome.opportunity = summary(
        profile=FakeProfile(kernels=[FakeKernel("<script>x</script>")]))
    html = html_report._repair_opportunity(outcome)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_the_summary_survives_the_json_round_trip():
    import json

    outcome = OptimizationResult(status="NO_REPAIR_AVAILABLE")
    outcome.opportunity = summary()
    payload = json.loads(json.dumps(outcome.to_dict()))["repair_opportunity"]
    assert payload["runtime_delegation"] == pytest.approx(0.343)
    assert payload["top_hotspot"]["operator"] == "_softmax.out"
    assert payload["theoretical_upper_bound_speedup"] == pytest.approx(2.732,
                                                                      abs=1e-3)


def test_the_json_carries_no_credential_shaped_field():
    import json

    outcome = OptimizationResult(status="NO_REPAIR_AVAILABLE")
    outcome.opportunity = summary(configuration=FakeConfiguration())
    text = json.dumps(outcome.to_dict()).lower()
    for forbidden in ("api_key", "secret", "token", "password"):
        assert forbidden not in text, forbidden


# --- the consent boundary ----------------------------------------------------

def test_the_consent_prompt_shows_the_decision_screen():
    said = []
    consent.request_repair_consent(summary(), interactive=True,
                                   preapproved=False,
                                   announce=said.append,
                                   prompt=lambda question: "n")
    text = "\n".join(said)
    assert "_softmax.out" in text
    assert "63.4%" in text


def test_the_flag_still_shows_the_numbers_it_is_spending_on():
    """--allow-ai pre-approves the request; it does not hide the context."""
    said = []
    decision = consent.request_repair_consent(
        summary(), interactive=False, preapproved=True, announce=said.append)
    assert decision.granted
    assert "_softmax.out" in "\n".join(said)


def test_the_answer_is_still_no_by_default():
    """More context must not have changed the default to yes."""
    decision = consent.request_repair_consent(
        summary(), interactive=True, preapproved=False,
        announce=lambda text: None, prompt=lambda question: "")
    assert not decision.granted


def test_a_bare_hotspot_string_still_renders_a_disclosure():
    """Callers without a profile keep working, and still disclose."""
    text = consent.graph_disclosure("aten.foo")
    assert "aten.foo" in text
    assert "It will NOT send:" in text
