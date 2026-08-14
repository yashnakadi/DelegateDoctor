"""The only place DelegateDoctor talks to an AI provider.

One narrow interface, so the rest of the project depends on an idea rather than
on a vendor:

    provider.complete_structured(request) -> AIResponse

What the provider is *not* given is the important part. There are no tools, no
function calling, no browsing, no filesystem access and no code execution - not
because the model is asked to behave, but because none of it is wired up. The
request carries a system instruction and one block of sanitized text, and the
reply is text that DelegateDoctor then validates against a schema.

Bring your own key
------------------
DelegateDoctor has no account, ships no key and funds no inference, and it
never stores a credential. Every request uses the local user's own key, read
from the environment at the moment of the call, passed to LiteLLM as an
argument rather than exported anywhere, and dropped afterwards. It is never in
argv, a log, an exception, a repr or an artifact.

LiteLLM is transport only. It normalizes providers; it does not decide what may
be sent, whether consent was given, or whether a reply is acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import credentials, privacy, provider_config, provider_response

LITELLM_VERSION = "1.96.2"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT_SECONDS = 120


class AIError(RuntimeError):
    """An AI request failed. The message is always sanitized."""


class AINotConfigured(AIError):
    """No provider credential is available."""


class AIDisabled(AIError):
    """AI was not permitted for this run.

    Raised when `build_provider(allow_ai=False)` is called, which is how a
    caller structurally rules AI out: there is nothing to make a request with,
    and no credential is even resolved.
    """


@dataclass
class AIRequest:
    """Everything that will be sent. Inspectable by tests, by construction."""

    system: str
    user: str
    max_tokens: int = DEFAULT_MAX_TOKENS
    purpose: str = ""          # "preparation" or "repair", for disclosure only

    def payload_text(self) -> str:
        """Every character that would leave this machine, for auditing."""
        return f"{self.system}\n{self.user}"


@dataclass
class AIResponse:
    text: str
    model: str = ""
    raw: dict = field(default_factory=dict, repr=False)


class Provider:
    """The interface the rest of DelegateDoctor depends on.

    Two methods, deliberately:

    `complete` reports what happened - success, refusal, empty, truncated,
    failed - and is what model-level repair exploration uses, because those
    outcomes need telling apart.

    `complete_structured` raises on anything but success. Preparation and
    export assistance want exactly that: for them a provider that did not
    answer is simply an error, and their retry loops are written around it.
    """

    name = "provider"

    def complete(self, request: AIRequest):
        """-> ProviderCompletionResult. Never raises for a provider outcome."""
        raise NotImplementedError

    def complete_structured(self, request: AIRequest) -> AIResponse:
        """-> AIResponse, or AIError for any non-success outcome."""
        result = self.complete(request)
        if not result.succeeded:
            raise AIError(_completion_error_text(result))
        return AIResponse(text=result.text,
                          model=getattr(self, "model", ""))


class LiteLLMProvider(Provider):
    """The transport. LiteLLM normalizes providers; it decides nothing else.

    Everything that matters - what may be sent, whether consent was given, what
    a valid reply looks like - has already happened by the time this is called.
    LiteLLM receives two finished strings and returns one.

    It is given no tools, no functions, no callbacks and no filesystem access.
    Its capability here is exactly: sanitized messages in, text out.
    """

    name = "litellm"

    def __init__(self, configuration, api_key: str = None,
                 timeout: int = DEFAULT_TIMEOUT_SECONDS, completion=None):
        self.configuration = configuration
        # Held only for this object's lifetime, which is one run. Never logged,
        # never in __repr__, never in an exception.
        self._api_key = api_key
        self.timeout = timeout
        self._completion = completion

    @property
    def model(self) -> str:
        return self.configuration.litellm_model

    def _call(self):
        if self._completion is not None:
            return self._completion
        try:
            import litellm
        except ImportError:
            raise AIError(
                "LiteLLM is not installed.\n"
                "\n"
                "    python -m pip install 'litellm==1.96.2'\n"
                "\n"
                "Or skip AI entirely:\n"
                "    from delegate_doctor import optimize\n"
                "    result = optimize(model.eval(), args=(example_input,))"
            )
        _configure_litellm_privacy(litellm)
        return litellm.completion

    def complete(self, request: AIRequest):
        """One provider call, reported rather than reduced to text-or-raise."""
        completion = self._call()

        arguments = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "max_tokens": request.max_tokens,
            "timeout": self.timeout,
            # JSON mode where the provider supports it. `drop_params` means a
            # provider that does not simply ignores it rather than failing -
            # and DelegateDoctor validates the reply either way, so this is an
            # optimization, never the guarantee.
            "response_format": {"type": "json_object"},
            # Explicitly no capability beyond text.
            "tools": None,
            "stream": False,
        }
        # Credentials are passed per request rather than exported into the
        # process environment, so nothing else in this process - or any child
        # of it - can read them.
        if self._api_key:
            arguments["api_key"] = self._api_key
        if getattr(self.configuration, "api_base", ""):
            arguments["api_base"] = self.configuration.api_base

        try:
            response = completion(**arguments)
        except Exception as error:
            translated = _translate_provider_error(error, self.configuration,
                                                   self._api_key)
            return provider_response.error_result(
                str(translated).splitlines()[0], exception=error,
                verbose_detail=str(translated)[:300])

        result = provider_response.extract_structured_provider_response(response)
        result.diagnostics.setdefault("provider", self.configuration.definition.label)
        result.diagnostics.setdefault("model", self.configuration.model)
        return result

    def __repr__(self) -> str:
        # Explicit, so no debugger frame or traceback can print the key.
        return f"<LiteLLMProvider model={self.model!r}>"


def _configure_litellm_privacy(litellm) -> None:
    """Turn off everything in LiteLLM that could move data or log secrets.

    LiteLLM has a large optional surface - verbose request logging, callbacks,
    third-party observability integrations. None of it is appropriate for a
    tool handling private model source, so it is switched off explicitly rather
    than left at whatever the default happens to be.
    """
    for attribute, value in (
        ("set_verbose", False),
        ("turn_off_message_logging", True),
        ("suppress_debug_info", True),
        ("telemetry", False),
        ("success_callback", []),
        ("failure_callback", []),
        ("callbacks", []),
        ("input_callback", []),
        ("service_callback", []),
        ("drop_params", True),
    ):
        try:
            setattr(litellm, attribute, value)
        except Exception:
            # A future version that renamed one of these is not a reason to
            # abort the run; the important ones are set, and DelegateDoctor
            # never passes tools or callbacks in the request itself.
            continue


def _completion_error_text(result) -> str:
    """The message `complete_structured` raises for a non-success outcome.

    Keeps the distinctions visible even to callers that only want an
    exception: "declined", "no content" and "ended before completing" lead
    somewhere different, and one sentence for all three is what hid a
    truncation behind the word "empty".
    """
    detail = result.message or result.reported_status
    return f"{result.reported_status}: {detail}"


def _translate_provider_error(error, configuration, api_key: str = None):
    """Turn a provider/LiteLLM failure into a short DelegateDoctor error.

    The text is sanitized: a provider error can echo a request header, and a
    stack trace from a vendor SDK is not something a user should have to read.
    """
    label = getattr(getattr(configuration, "definition", None), "label",
                    "your AI provider")
    name = type(error).__name__
    detail = str(error)
    # Belt and braces: a provider error can echo the credential back verbatim
    # in a form no generic pattern recognises (`key=...`, a URL fragment, a
    # quoted header). The exact value is known here, so remove it by identity
    # before the generic redaction runs.
    if api_key:
        detail = detail.replace(api_key, privacy.PLACEHOLDER)
    detail = privacy.redact(detail)[:300]
    lowered = f"{name} {detail}".lower()

    if "auth" in lowered or "api key" in lowered or "401" in lowered:
        return AIError(
            f"AI PROVIDER AUTHENTICATION FAILED\n"
            f"\n"
            f"{label} rejected the credential DelegateDoctor supplied.\n"
            f"\n"
            f"Check or replace it with:\n"
            f"    delegate-doctor configure-ai"
        )
    if "rate" in lowered and "limit" in lowered:
        return AIError(
            f"AI PROVIDER RATE LIMIT\n"
            f"\n"
            f"{label} is rate limiting this account. Wait and retry."
        )
    if "quota" in lowered or "billing" in lowered or "credit" in lowered:
        return AIError(
            f"AI PROVIDER QUOTA OR BILLING PROBLEM\n"
            f"\n"
            f"{label} refused the request for billing reasons. DelegateDoctor "
            f"uses your own provider account."
        )
    if "context" in lowered and ("length" in lowered or "window" in lowered):
        return AIError(
            f"AI MODEL CONTEXT LIMIT EXCEEDED\n"
            f"\n"
            f"The request was too large for this model. Choose a larger one:\n"
            f"    delegate-doctor configure-ai"
        )
    if "response_format" in lowered or "json" in lowered and "support" in lowered:
        return AIError(
            f"AI MODEL DOES NOT SUPPORT REQUIRED STRUCTURED OUTPUT\n"
            f"\n"
            f"Choose another model with:\n"
            f"    delegate-doctor configure-ai"
        )
    if "connect" in lowered or "timeout" in lowered or "unavailable" in lowered:
        return AIError(
            f"AI PROVIDER UNAVAILABLE\n"
            f"\n"
            f"Could not reach {label}."
        )
    return AIError(f"The AI request failed ({name}).\n{detail}")


def build_provider(allow_ai: bool = True, configuration=None) -> Provider:
    """Construct the configured provider, or explain why there is none.

    With `allow_ai=False` this raises before a provider object exists, which is
    what makes refusal structural rather than a policy: there is nothing to
    make a request with, and no credential is even resolved.
    """
    if not allow_ai:
        raise AIDisabled(
            "AI is not permitted for this run.\n"
            "\n"
            "Export the model yourself to run the deterministic path:\n"
            "\n"
            "    from delegate_doctor import optimize\n"
            "    result = optimize(model.eval(), args=(example_input,))"
        )

    configuration = configuration or provider_config.load_configuration()
    if configuration is None:
        raise AINotConfigured(credentials.NOT_CONFIGURED_MESSAGE)

    api_key = None
    if configuration.needs_api_key:
        api_key = credentials.resolve_api_key(configuration.provider)
        if not api_key:
            raise AINotConfigured(
                credentials.missing_key_message(configuration))

    return LiteLLMProvider(configuration=configuration, api_key=api_key)
