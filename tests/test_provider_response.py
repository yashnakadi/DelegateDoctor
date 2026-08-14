"""Telling apart what actually happened after the provider call.

The bug: a real Inception run reached the provider, and reported

    AI-CANDIDATE-001
    Result   NO_CANDIDATE
    Detail   provider error: The AI provider returned an empty response.

Two things are wrong there. A candidate was invented for a call that returned
nothing, and "empty" was the only diagnosis available - because extraction read
`message.content` and nothing else, so a refusal, a truncation and genuine
silence were indistinguishable.

These tests hold the distinction: a provider call is not a repair candidate,
and each outcome is named.

The response shapes below match the installed LiteLLM (1.96.2): `Choices` has
`finish_reason`, `Message` has `content`/`tool_calls`/`reasoning_content`, and
`Message` is `extra="allow"` so an OpenAI `refusal` key survives as an extra.

Fully offline: no provider, no network, no device.
"""

import json

import pytest

from delegate_doctor.agent import provider_response
from delegate_doctor.agent.client import AIError, AIRequest, LiteLLMProvider
from delegate_doctor.agent.provider_response import (EMPTY, ERROR, INVALID,
                                                     REFUSED, SUCCESS)


# --- response doubles, shaped like LiteLLM's ------------------------------------

class Message:
    def __init__(self, content=None, refusal=None, tool_calls=None,
                 reasoning_content=None):
        self.content = content
        self.refusal = refusal
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


class Choice:
    def __init__(self, message, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason


class Usage:
    def __init__(self, prompt=100, completion=50, reasoning=None):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion
        self.completion_tokens_details = (
            type("Details", (), {"reasoning_tokens": reasoning})()
            if reasoning is not None else None)


class Response:
    def __init__(self, choice=None, usage=None):
        self.choices = [choice] if choice is not None else []
        self.usage = usage


PLAN = json.dumps({
    "summary": "insert a clone",
    "anchor": "node_1",
    "operations": [{"type": "insert_aten_call", "id": "new_1",
                    "target": "aten.clone.default",
                    "args": [{"node": "node_1"}], "before": "node_2"}],
})


# --- extraction: one function, deterministic order --------------------------------

def test_textual_content_is_a_success():
    """Case 2: a valid structured response arrives as content."""
    result = provider_response.extract_structured_provider_response(
        Response(Choice(Message(content=PLAN))))
    assert result.status == SUCCESS
    assert result.succeeded
    assert result.text == PLAN
    assert result.reported_status == "REPAIR_PROPOSALS_RETURNED"


def test_an_explicit_refusal_is_not_an_empty_response():
    """Case 5: the model declined. That is an answer."""
    result = provider_response.extract_structured_provider_response(
        Response(Choice(Message(refusal="I can't help with that"))))
    assert result.status == REFUSED
    assert result.reported_status == "PROVIDER_REFUSED"
    assert "declined" in result.message


def test_content_wins_over_a_refusal_field():
    """A response carrying both said something usable."""
    result = provider_response.extract_structured_provider_response(
        Response(Choice(Message(content=PLAN, refusal="partial concern"))))
    assert result.status == SUCCESS


def test_truncation_is_not_an_empty_response():
    """Case 8: output existed and the budget ran out. Say so."""
    result = provider_response.extract_structured_provider_response(
        Response(Choice(Message(content=""), finish_reason="length")))
    assert result.status == INVALID
    assert result.reported_status == "INVALID_STRUCTURED_RESPONSE"
    assert "ended before" in result.message
    assert result.diagnostics["finish reason"] == "length"


def test_genuinely_empty_content_is_empty():
    """Case 4."""
    result = provider_response.extract_structured_provider_response(
        Response(Choice(Message(content=None))))
    assert result.status == EMPTY
    assert result.reported_status == "PROVIDER_EMPTY_RESPONSE"


def test_whitespace_only_content_is_empty():
    result = provider_response.extract_structured_provider_response(
        Response(Choice(Message(content="   \n  "))))
    assert result.status == EMPTY


def test_a_response_with_no_choices_is_empty():
    result = provider_response.extract_structured_provider_response(Response())
    assert result.status == EMPTY
    assert "no choices" in result.message


def test_a_dict_shaped_response_is_read_the_same_way():
    """LiteLLM normalizes, but a plain dict must not crash extraction."""
    result = provider_response.extract_structured_provider_response(
        {"choices": [{"message": {"content": PLAN}, "finish_reason": "stop"}]})
    assert result.status == SUCCESS
    assert result.text == PLAN


def test_extraction_lives_in_exactly_one_place():
    """Case 3: no field probing scattered through the explorer."""
    import inspect

    from delegate_doctor.agent import repair_explorer

    source = inspect.getsource(repair_explorer)
    for probe in ("choices[0]", ".message.content", 'response["choices"]'):
        assert probe not in source, f"repair_explorer probes {probe} itself"


# --- diagnostics are structural, never content ---------------------------------------

def test_diagnostics_describe_the_response_without_quoting_it():
    """Cases 15/16: no key, no prompt, no model output."""
    secret = "sk-super-secret-value"
    response = Response(Choice(Message(content="PRIVATE MODEL GRAPH CONTENT")),
                        usage=Usage())
    diagnostics = provider_response.describe_response(response)

    text = json.dumps(diagnostics)
    assert secret not in text
    assert "PRIVATE MODEL GRAPH CONTENT" not in text
    assert diagnostics["content present"] == "YES"
    assert diagnostics["request completed"] == "YES"
    assert diagnostics["finish reason"] == "stop"


def test_diagnostics_report_token_usage():
    diagnostics = provider_response.describe_response(
        Response(Choice(Message(content="x")), usage=Usage(prompt=10,
                                                           completion=20)))
    assert diagnostics["prompt tokens"] == "10"
    assert diagnostics["output tokens"] == "20"


def test_reasoning_tokens_are_reported_when_present():
    """A reasoning model can spend the whole budget before emitting content."""
    diagnostics = provider_response.describe_response(
        Response(Choice(Message(content=""), finish_reason="length"),
                 usage=Usage(completion=2048, reasoning=2048)))
    assert diagnostics["reasoning tokens"] == "2048"
    assert diagnostics["content present"] == "NO"


def test_no_raw_repr_of_the_response_is_produced():
    """A repr would carry whatever the model echoed back.

    Comments and docstrings are stripped first: this module *explains* why it
    does not do this, and a plain text scan cannot tell a prohibition from an
    occurrence.
    """
    import inspect
    import io
    import tokenize

    source = inspect.getsource(provider_response)
    kept = []
    previous = tokenize.INDENT
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            continue
        if token.type == tokenize.STRING and previous in (
                tokenize.INDENT, tokenize.NEWLINE, tokenize.NL,
                tokenize.DEDENT):
            previous = token.type
            continue
        if token.type not in (tokenize.NL, tokenize.NEWLINE):
            previous = token.type
        kept.append(token.string)
    code = "".join(kept)

    assert "repr(response)" not in code
    assert "str(response)" not in code


def test_the_outcome_block_is_terse_without_verbose():
    result = provider_response.ProviderCompletionResult(
        EMPTY, message="the provider returned no content",
        diagnostics={"finish reason": "stop", "content present": "NO"})

    terse = provider_response.format_outcome(result, "OpenAI · gpt-4o",
                                             verbose=False)
    assert "PROVIDER_EMPTY_RESPONSE" in terse
    assert "finish reason" not in terse.lower() or "Finish reason" not in terse

    detailed = provider_response.format_outcome(result, "OpenAI · gpt-4o",
                                                verbose=True)
    assert "Finish reason" in detailed
    assert "Content present" in detailed


# --- the provider wrapper ----------------------------------------------------------------

class Configuration:
    class definition:
        label = "OpenAI"

    model = "gpt-4o"
    api_base = ""

    @property
    def litellm_model(self):
        return "openai/gpt-4o"

    def describe(self):
        return "OpenAI · gpt-4o"


def provider_with(completion):
    return LiteLLMProvider(configuration=Configuration(), api_key="sk-test",
                           completion=completion)


def test_a_provider_exception_is_an_error_not_a_candidate():
    """Case 6."""
    def explode(**kwargs):
        raise RuntimeError("rate limit exceeded")

    result = provider_with(explode).complete(AIRequest(system="s", user="u"))
    assert result.status == ERROR
    assert result.reported_status == "PROVIDER_ERROR"
    assert result.diagnostics["request completed"] == "NO"


def test_a_provider_error_never_leaks_the_key():
    """Case 15."""
    def explode(**kwargs):
        raise RuntimeError("auth failed for key sk-test")

    result = provider_with(explode).complete(AIRequest(system="s", user="u"))
    assert "sk-test" not in json.dumps(result.to_dict())
    assert "sk-test" not in result.message


def test_complete_structured_still_raises_for_the_other_callers():
    """Preparation and export assistance are written around an exception."""
    result = Response(Choice(Message(content=None)))
    with pytest.raises(AIError) as caught:
        provider_with(lambda **kwargs: result).complete_structured(
            AIRequest(system="s", user="u"))
    # And the exception names the outcome rather than flattening it.
    assert "PROVIDER_EMPTY_RESPONSE" in str(caught.value)


def test_complete_structured_returns_text_on_success():
    response = Response(Choice(Message(content=PLAN)))
    reply = provider_with(lambda **kwargs: response).complete_structured(
        AIRequest(system="s", user="u"))
    assert reply.text == PLAN


def test_the_request_configuration_is_unchanged():
    """Case 9/17: nothing about the outbound request moved."""
    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return Response(Choice(Message(content=PLAN)))

    provider_with(capture).complete(AIRequest(system="s", user="u"))

    assert seen["response_format"] == {"type": "json_object"}
    assert seen["tools"] is None
    assert seen["stream"] is False
    assert "temperature" not in seen
    assert seen["api_key"] == "sk-test"
    assert seen["max_tokens"] > 0


# --- the explorer: a provider failure is not a candidate ------------------------------------

class ScriptedProvider:
    configuration = Configuration()

    def __init__(self, *results):
        self.results = list(results)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return self.results.pop(0) if self.results else \
            provider_response.ProviderCompletionResult(EMPTY)


def explore_with(provider, known=("node_0", "node_1", "node_2")):
    from delegate_doctor.agent import repair_explorer

    return repair_explorer.explore(
        provider=provider, baseline_program=None, context={"graph": {}},
        known_nodes=list(known), lower=lambda program: None,
        announce=lambda line: None)


def test_a_provider_failure_produces_no_candidate_attempts():
    """Case 9/12: nothing was proposed, so nothing is counted."""
    provider = ScriptedProvider(
        provider_response.ProviderCompletionResult(
            EMPTY, message="the provider returned no content"))
    result = explore_with(provider)

    assert result.attempts == []
    assert result.candidates_proposed == 0
    assert not result.provider_succeeded
    assert result.provider_result.status == EMPTY


def test_a_provider_failure_stops_after_one_request():
    """No point re-asking a provider that could not answer."""
    provider = ScriptedProvider(
        provider_response.ProviderCompletionResult(ERROR, message="timeout"),
        provider_response.ProviderCompletionResult(SUCCESS, text=PLAN))
    explore_with(provider)
    assert len(provider.requests) == 1


def test_an_explicit_decline_is_a_successful_response():
    """Case 3: NO_REPAIR_PROPOSED, not an error."""
    provider = ScriptedProvider(
        provider_response.ProviderCompletionResult(
            SUCCESS, text='{"no_repair": true, "reason": "nothing safe"}'))
    result = explore_with(provider)

    assert result.declined
    assert result.provider_succeeded
    assert result.attempts == []
    assert result.candidates_proposed == 0


@pytest.mark.parametrize("text", [
    '{"anchor": "node_1", "operations": []}',
    "prose, not a plan",
    '{"no_repair": false}',
])
def test_malformed_content_is_a_recorded_proposal_not_a_provider_failure(text):
    """Case 7: content arrived. The schema refused it. That is a proposal."""
    provider = ScriptedProvider(
        provider_response.ProviderCompletionResult(SUCCESS, text=text),
        provider_response.ProviderCompletionResult(SUCCESS, text=text))
    result = explore_with(provider)

    assert result.provider_succeeded
    assert result.candidates_proposed >= 1
    assert all(attempt.outcome == "invalid" for attempt in result.attempts)


def test_a_valid_proposal_is_counted_once():
    """Case 10."""
    from delegate_doctor.agent import repair_explorer

    applied = []

    def fake_apply(baseline, plan):
        applied.append(plan)
        return "rewritten-program"

    import delegate_doctor.agent.repair_explorer as module
    original = module.apply_candidate
    module.apply_candidate = fake_apply
    try:
        provider = ScriptedProvider(
            provider_response.ProviderCompletionResult(SUCCESS, text=PLAN))
        result = explore_with(provider)
    finally:
        module.apply_candidate = original

    assert result.candidates_proposed == 1
    assert result.found_runnable
    assert len(result.runnable_candidates) == 1


def test_bounded_retries_survive_for_malformed_output():
    """Case 6/I: the existing retry policy is untouched."""
    from delegate_doctor.agent.repair_explorer import MAX_AI_REPAIR_CANDIDATES

    provider = ScriptedProvider(*[
        provider_response.ProviderCompletionResult(SUCCESS, text="garbage")
        for _ in range(MAX_AI_REPAIR_CANDIDATES)])
    result = explore_with(provider)

    assert len(provider.requests) == MAX_AI_REPAIR_CANDIDATES
    assert result.candidates_proposed == MAX_AI_REPAIR_CANDIDATES


def test_the_model_level_request_happens_once_per_exploration():
    """Case 19: one request for the model, not one per hotspot."""
    provider = ScriptedProvider(
        provider_response.ProviderCompletionResult(
            SUCCESS, text='{"no_repair": true}'))
    explore_with(provider)
    assert len(provider.requests) == 1


# --- stale hotspot-era wording ------------------------------------------------------------

def test_the_consent_screen_is_model_level_not_hotspot_level():
    """Case 20."""
    import inspect
    from pathlib import Path

    from delegate_doctor import repair_opportunity

    package = Path(inspect.getfile(repair_opportunity)).parent
    for path in package.rglob("*.py"):
        text = path.read_text()
        for stale in ("matches the measured hotspot",
                      "sanitized neighbourhood of this"):
            assert stale not in text, f"{path.name} still says {stale!r}"


def test_the_disclosure_describes_the_exported_graph():
    from delegate_doctor import repair_opportunity

    text = repair_opportunity.format_decision_screen(
        repair_opportunity.build_summary())
    assert "No known DelegateDoctor repairs remain." in text
    assert "experimentally inspect the measured exported" in text
    # The privacy enumeration is unchanged.
    assert "It will NOT send:" in text


def test_a_decline_is_reported_under_its_own_name():
    """Not REPAIR_PROPOSALS_RETURNED: no proposals were made, deliberately."""
    declined = provider_response.ProviderCompletionResult(
        SUCCESS, message="provider found no safe DSL-expressible repair")
    declined.status = provider_response.NO_REPAIR_PROPOSED

    assert declined.declined
    assert declined.reported_status == "NO_REPAIR_PROPOSED"
    block = provider_response.format_outcome(declined, "OpenAI · gpt-4o")
    assert block.count("Result ") == 1
    assert "REPAIR_PROPOSALS_RETURNED" not in block
