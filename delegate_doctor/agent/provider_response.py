"""What actually came back from the provider, told apart properly.

A provider call has several genuinely different outcomes, and DelegateDoctor
used to collapse all of them into one line:

    AI-CANDIDATE-001
    Result   NO_CANDIDATE
    Detail   provider error: The AI provider returned an empty response.

That sentence was produced by `message.content` being empty - and *only* by
that. A refusal, a response truncated at the token limit, and a model that
never emitted anything all arrived here identically, as did the invention of a
"candidate" that had never existed.

So this module answers one question, once:

    SUCCESS      a usable structured payload came back
    REFUSED      the model explicitly declined
    EMPTY        the request completed and produced no output at all
    INVALID      output arrived but is not a usable structured payload
    ERROR        the call itself failed - network, auth, rate limit, timeout

Only SUCCESS can lead to a repair candidate. The rest are facts about the
request, and none of them is a candidate.

Field probing lives here and nowhere else. The shapes below were read from the
installed LiteLLM (1.96.2), not assumed:

    Choices.finish_reason                       exists
    Message.content                             exists
    Message.tool_calls, .reasoning_content      exist
    Message.refusal                             NOT a declared field

`Message` is configured `extra="allow"`, so a `refusal` key returned by OpenAI
survives as an extra attribute even though LiteLLM does not model it. It is
therefore read best-effort, and its absence is never mistaken for its being
false.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- outcomes -------------------------------------------------------------------

SUCCESS = "SUCCESS"
REFUSED = "REFUSED"
EMPTY = "EMPTY"
INVALID = "INVALID"
ERROR = "ERROR"

# How each outcome is named where a user reads it.
REPORTED_STATUS = {
    SUCCESS: "REPAIR_PROPOSALS_RETURNED",
    REFUSED: "PROVIDER_REFUSED",
    EMPTY: "PROVIDER_EMPTY_RESPONSE",
    INVALID: "INVALID_STRUCTURED_RESPONSE",
    ERROR: "PROVIDER_ERROR",
}

# A successful response that deliberately proposes nothing. Distinct from EMPTY:
# the model answered, and its answer was "no safe repair".
NO_REPAIR_PROPOSED = "NO_REPAIR_PROPOSED"

# A proposal arrived and could not be turned into a runnable graph. That is a
# repair outcome, not a provider outcome, and it is named here only so the
# vocabulary lives in one place.
NO_RUNNABLE_CANDIDATE = "NO_RUNNABLE_CANDIDATE"


@dataclass
class ProviderCompletionResult:
    """One provider call, and precisely what became of it."""

    status: str
    text: str = ""
    message: str = ""
    diagnostics: dict = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == SUCCESS

    @property
    def declined(self) -> bool:
        """A successful call whose answer was "no safe repair"."""
        return self.status == NO_REPAIR_PROPOSED

    @property
    def reported_status(self) -> str:
        return REPORTED_STATUS.get(self.status, self.status)

    def describe(self) -> str:
        return self.message or self.reported_status

    def to_dict(self) -> dict:
        return {"status": self.status, "reported": self.reported_status,
                "message": self.message, "diagnostics": dict(self.diagnostics)}


# --- extraction --------------------------------------------------------------------


def _first_choice(response):
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        return None
    return choices[0]


def _attribute(holder, name):
    """Read a field from a pydantic model or a plain dict, without raising."""
    if holder is None:
        return None
    value = getattr(holder, name, None)
    if value is None and isinstance(holder, dict):
        value = holder.get(name)
    return value


def describe_response(response) -> dict:
    """Structural metadata only. Never content, never the prompt, never a key.

    Everything here is a type name, a boolean, a finish reason or a token
    count. Deliberately no `repr(response)`: the response object carries the
    model's output, and printing it wholesale to a terminal would leak whatever
    the model happened to echo back.
    """
    choice = _first_choice(response)
    message = _attribute(choice, "message")
    content = _attribute(message, "content")
    refusal = _attribute(message, "refusal")
    tool_calls = _attribute(message, "tool_calls")
    reasoning = _attribute(message, "reasoning_content")
    usage = _attribute(response, "usage")

    diagnostics = {
        "request completed": "YES",
        "response type": type(response).__name__,
        "finish reason": str(_attribute(choice, "finish_reason") or "unknown"),
        "content present": "YES" if (content or "").strip() else "NO",
        "refusal": "YES" if (refusal or "").strip() else "NO",
        "tool calls": "YES" if tool_calls else "NO",
        "reasoning content": "YES" if (reasoning or "").strip() else "NO",
    }
    if usage is not None:
        for label, name in (("prompt tokens", "prompt_tokens"),
                            ("output tokens", "completion_tokens"),
                            ("total tokens", "total_tokens")):
            value = _attribute(usage, name)
            if value is not None:
                diagnostics[label] = str(value)
        # Reasoning models spend the output budget before emitting content.
        # When that is what happened, this is the number that explains it.
        details = _attribute(usage, "completion_tokens_details")
        reasoning_tokens = _attribute(details, "reasoning_tokens")
        if reasoning_tokens is not None:
            diagnostics["reasoning tokens"] = str(reasoning_tokens)
    return diagnostics


def extract_structured_provider_response(response) -> ProviderCompletionResult:
    """The one place a provider response is interpreted.

    Checked in a deterministic order, and only against fields the installed
    LiteLLM actually produces:

        1. textual content            -> SUCCESS
        2. an explicit refusal        -> REFUSED
        3. truncated before content   -> INVALID, named as truncation
        4. anything else              -> EMPTY

    Refusal is checked *after* content because a response that carries both has
    said something usable, and before the empty case because "the model
    declined" and "the model produced nothing" are different answers.
    """
    diagnostics = describe_response(response)

    choice = _first_choice(response)
    if choice is None:
        return ProviderCompletionResult(
            EMPTY, message="the provider returned no choices",
            diagnostics=diagnostics)

    message = _attribute(choice, "message")
    content = (_attribute(message, "content") or "").strip()
    if content:
        return ProviderCompletionResult(SUCCESS, text=content,
                                        diagnostics=diagnostics)

    refusal = (_attribute(message, "refusal") or "").strip()
    if refusal:
        return ProviderCompletionResult(
            REFUSED, message="the model declined to answer",
            diagnostics=diagnostics)

    finish_reason = str(_attribute(choice, "finish_reason") or "").lower()
    if finish_reason == "length":
        # Output existed - the budget ran out before it was complete. Calling
        # this "empty" hid the one thing that would have explained it.
        return ProviderCompletionResult(
            INVALID,
            message="generation ended before a structured response completed",
            diagnostics=diagnostics)

    return ProviderCompletionResult(
        EMPTY, message="the provider returned no content",
        diagnostics=diagnostics)


def error_result(reason: str, exception: Exception = None,
                 verbose_detail: str = "") -> ProviderCompletionResult:
    """A call that did not complete: network, auth, rate limit, timeout."""
    diagnostics = {"request completed": "NO"}
    if exception is not None:
        diagnostics["exception"] = type(exception).__name__
    if verbose_detail:
        diagnostics["detail"] = verbose_detail
    return ProviderCompletionResult(ERROR, message=reason,
                                    diagnostics=diagnostics)


def invalid_result(reason: str, previous: ProviderCompletionResult = None
                   ) -> ProviderCompletionResult:
    """Content arrived and the schema refused it."""
    diagnostics = dict(previous.diagnostics) if previous else {}
    diagnostics["structured result"] = "NO"
    return ProviderCompletionResult(INVALID, message=reason,
                                    diagnostics=diagnostics)


# --- reporting -----------------------------------------------------------------------


def format_outcome(result: ProviderCompletionResult, provider_label: str = "",
                   verbose: bool = False) -> str:
    """The terminal block for one exploration's provider outcome."""
    lines = ["", "AI exploration"]
    if provider_label:
        lines.append(f"Provider                {provider_label}")
    lines.append(f"Result                  {result.reported_status}")
    if result.message and result.status != SUCCESS:
        lines.append(f"Detail                  {result.message}")
    if verbose and result.diagnostics:
        for key, value in result.diagnostics.items():
            lines.append(f"{key.capitalize():<24}{value}")
    return "\n".join(lines)
