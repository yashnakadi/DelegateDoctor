"""Bring-your-own-key: provider selection, and the LiteLLM transport boundary.

Two layers, deliberately separate. Domain tests elsewhere use `FakeProvider`
and never touch LiteLLM; these mock `litellm.completion` itself and check what
DelegateDoctor hands it - the model string, the credential, the absence of
tools, and the absence of anything private.
"""

import json

import pytest

from delegate_doctor.agent import client, credentials, provider_config
from delegate_doctor.agent.client import AIError, AIRequest, LiteLLMProvider
from delegate_doctor.agent.provider_config import ProviderConfigError
# Captured at import time: conftest deliberately blocks the module attribute.
from delegate_doctor.agent.client import build_provider as REAL_BUILD_PROVIDER

SENTINEL = "DD_TEST_SUPER_SECRET_8f34c1d9e7b2a5"


class FakeCompletion:
    """Stands in for litellm.completion, recording exactly what it received."""

    def __init__(self, text='{"ok": true}', error=None):
        self.text = text
        self.error = error
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error

        class Message:
            content = self.text

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        return Response()



def _code_without_prose(path: str) -> str:
    """Module source with comments and every docstring removed.

    These modules describe the things they refuse to do, so a plain text scan
    cannot tell a refusal from an occurrence.
    """
    import io
    import tokenize

    kept = []
    previous = tokenize.INDENT
    with open(path) as handle:
        for token in tokenize.generate_tokens(handle.readline):
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
    return "\n".join(kept)


def provider_for(key="anthropic", api_key=SENTINEL, completion=None, **extra):
    configuration = provider_config.build_configuration(key, **extra)
    return LiteLLMProvider(configuration=configuration, api_key=api_key,
                           completion=completion or FakeCompletion())


# --- provider selection --------------------------------------------------------

@pytest.mark.parametrize("key, expected_prefix", [
    ("openai", "openai/"),
    ("anthropic", "anthropic/"),
    ("gemini", "gemini/"),
    ("openrouter", "openrouter/"),
    ("ollama", "ollama_chat/"),
])
def test_each_first_class_provider_qualifies_its_model(key, expected_prefix):
    configuration = provider_config.build_configuration(key)
    assert configuration.litellm_model.startswith(expected_prefix)


def test_an_advanced_litellm_model_string_is_preserved():
    configuration = provider_config.build_configuration(
        "advanced", "azure/my-deployment")
    assert configuration.litellm_model == "azure/my-deployment"


def test_an_already_qualified_model_is_not_double_prefixed():
    configuration = provider_config.build_configuration(
        "anthropic", "anthropic/claude-sonnet-4-5")
    assert configuration.litellm_model == "anthropic/claude-sonnet-4-5"


def test_an_unknown_provider_is_rejected():
    with pytest.raises(ProviderConfigError) as caught:
        provider_config.build_configuration("definitely-not-a-provider")
    assert "Unknown provider" in str(caught.value)


@pytest.mark.parametrize("model", [
    "", "   ", "x" * 500, "model\nAuthorization: Bearer abc",
    "model\x00null", "https://user:secret@host/model",
])
def test_unsafe_model_strings_are_rejected(model):
    with pytest.raises(ProviderConfigError):
        provider_config.validate_model(model)


def test_the_onboarding_menu_stays_short():
    """LiteLLM supports a hundred providers; the menu must not."""
    assert len(provider_config.PROVIDERS) <= 8
    menu = provider_config.describe_menu()
    for label in ("OpenAI", "Anthropic", "Gemini", "OpenRouter", "Ollama"):
        assert label in menu


# --- local versus remote --------------------------------------------------------

def test_ollama_is_local():
    configuration = provider_config.build_configuration("ollama")
    assert configuration.is_local
    assert configuration.processing_label == provider_config.PROCESSING_LOCAL
    assert not configuration.needs_api_key


@pytest.mark.parametrize("key", ["openai", "anthropic", "gemini", "openrouter"])
def test_remote_providers_are_remote_and_need_a_key(key):
    configuration = provider_config.build_configuration(key)
    assert not configuration.is_local
    assert configuration.processing_label == provider_config.PROCESSING_REMOTE
    assert configuration.needs_api_key


def test_a_loopback_api_base_is_still_local():
    configuration = provider_config.build_configuration(
        "ollama", api_base="http://127.0.0.1:11434")
    assert configuration.is_local


def test_a_remote_api_base_is_not_assumed_local():
    """A friendly name is not proof. Being wrong here would be a false promise."""
    configuration = provider_config.build_configuration(
        "ollama", api_base="https://ollama.example.com")
    assert not configuration.is_local
    assert configuration.processing_label == provider_config.PROCESSING_REMOTE


def test_an_advanced_provider_is_never_assumed_local():
    configuration = provider_config.build_configuration("advanced", "azure/x")
    assert not configuration.is_local


# --- configuration persistence ---------------------------------------------------

def test_configuration_round_trips_without_a_secret(tmp_path):
    configuration = provider_config.build_configuration("openai", "gpt-4o")
    path = provider_config.save_configuration(configuration, tmp_path / "ai.json")

    payload = json.loads(path.read_text())
    assert payload == {"provider": "openai", "model": "gpt-4o"}

    loaded = provider_config.load_configuration(path)
    assert loaded.provider == "openai"
    assert loaded.model == "gpt-4o"


def test_a_missing_or_broken_config_returns_none(tmp_path):
    assert provider_config.load_configuration(tmp_path / "absent.json") is None
    broken = tmp_path / "ai.json"
    broken.write_text("{not json")
    assert provider_config.load_configuration(broken) is None


def test_the_config_file_refuses_secret_shaped_fields(tmp_path, monkeypatch):
    configuration = provider_config.build_configuration("openai")
    monkeypatch.setattr(configuration.__class__, "to_dict",
                        lambda self: {"provider": "openai", "api_key": SENTINEL})
    with pytest.raises(ProviderConfigError) as caught:
        provider_config.save_configuration(configuration, tmp_path / "ai.json")
    assert "does not store credentials" in str(caught.value)
    assert not (tmp_path / "ai.json").exists()


# --- credential resolution: environment only -------------------------------------

@pytest.fixture(autouse=True)
def clean_credential_environment(monkeypatch):
    """No inherited credential may influence these tests."""
    for name in (credentials.ENVIRONMENT_VARIABLE, "OPENAI_API_KEY",
                 "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("provider, variable", [
    ("openai", "OPENAI_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("gemini", "GEMINI_API_KEY"),
    ("openrouter", "OPENROUTER_API_KEY"),
])
def test_the_generic_variable_wins_over_the_provider_variable(
        provider, variable, monkeypatch):
    monkeypatch.setenv(credentials.ENVIRONMENT_VARIABLE, "generic-value")
    monkeypatch.setenv(variable, "provider-value")
    assert credentials.resolve_api_key(provider) == "generic-value"


@pytest.mark.parametrize("provider, variable", [
    ("openai", "OPENAI_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("gemini", "GEMINI_API_KEY"),
    ("openrouter", "OPENROUTER_API_KEY"),
])
def test_the_provider_variable_is_the_compatibility_fallback(
        provider, variable, monkeypatch):
    monkeypatch.setenv(variable, SENTINEL)
    assert credentials.resolve_api_key(provider) == SENTINEL


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini",
                                      "openrouter"])
def test_no_environment_variable_means_no_credential(provider):
    assert credentials.resolve_api_key(provider) is None


def test_one_providers_variable_does_not_satisfy_another(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL)
    assert credentials.resolve_api_key("anthropic") is None


def test_the_generic_variable_does_not_choose_the_provider(monkeypatch, tmp_path):
    """It is a credential, not a selector. Provider comes from config."""
    monkeypatch.setenv(credentials.ENVIRONMENT_VARIABLE, SENTINEL)
    configuration = provider_config.build_configuration("anthropic")
    path = provider_config.save_configuration(configuration, tmp_path / "ai.json")
    loaded = provider_config.load_configuration(path)
    assert loaded.provider == "anthropic"
    assert loaded.litellm_model.startswith("anthropic/")


def test_the_key_shape_is_never_inspected():
    """No prefix sniffing: an OpenAI-looking key for Anthropic is the user's call.

    Checked against code with the module docstring removed - the prose there
    explains that DelegateDoctor deliberately does not guess from a prefix.
    """
    code = _code_without_prose(credentials.__file__)
    for guess in ("sk-ant", 'startswith("sk-', "prefix"):
        assert guess not in code


def test_a_local_provider_needs_no_credential():
    configuration = provider_config.build_configuration("ollama")
    status = credentials.key_status(configuration)
    assert status.available
    assert not status.required
    assert "Not required" in status.describe()


def test_status_is_provider_aware(monkeypatch):
    anthropic = provider_config.build_configuration("anthropic")
    assert not credentials.key_status(anthropic).available

    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)
    status = credentials.key_status(anthropic)
    assert status.available
    assert status.source == "ANTHROPIC_API_KEY environment variable"


def test_the_generic_variable_is_reported_as_the_source(monkeypatch):
    monkeypatch.setenv(credentials.ENVIRONMENT_VARIABLE, SENTINEL)
    status = credentials.key_status(provider_config.build_configuration("openai"))
    assert status.source == credentials.SOURCE_GENERIC


def test_status_never_reveals_any_part_of_the_key(monkeypatch):
    monkeypatch.setenv(credentials.ENVIRONMENT_VARIABLE, SENTINEL)
    status = credentials.key_status(
        provider_config.build_configuration("anthropic"))
    rendered = status.describe() + repr(status) + str(status)
    assert SENTINEL not in rendered
    assert SENTINEL[:8] not in rendered          # not even a prefix
    assert SENTINEL[-4:] not in rendered         # not even the last four


# --- nothing stores a credential ---------------------------------------------------

def test_the_credentials_module_exposes_no_storage_api():
    """The security claim is structural: there is no function to store a key."""
    for removed in ("store_api_key", "delete_api_key", "keyring_available",
                    "_keyring", "KEYRING_SERVICE", "KEYRING_USERNAME"):
        assert not hasattr(credentials, removed), \
            f"credentials still exposes {removed}"


def test_no_module_imports_keyring():
    from pathlib import Path

    root = Path(credentials.__file__).parent.parent
    for path in root.rglob("*.py"):
        text = path.read_text()
        for token in ("import keyring", "keyring.get_password",
                      "keyring.set_password", "keyring.delete_password"):
            assert token not in text, f"{path.name} still uses keyring"


def test_keyring_is_not_a_dependency():
    from pathlib import Path

    text = (Path(credentials.__file__).parent.parent.parent
            / "pyproject.toml").read_text()
    assert "keyring" not in text


def test_configure_ai_never_reads_a_secret():
    """No getpass, and no prompt that could receive a key."""
    code = _code_without_prose(credentials.__file__)
    assert "getpass" not in code
    assert "prompt_for_secret" not in code


def test_nothing_writes_a_shell_profile():
    from pathlib import Path

    root = Path(credentials.__file__).parent.parent
    for path in root.rglob("*.py"):
        text = path.read_text()
        for profile in (".zshrc", ".bashrc", ".bash_profile", "setx ",
                        "PowerShell profile"):
            assert profile not in text, f"{path.name} touches {profile}"


def test_nothing_reads_or_writes_dotenv():
    from pathlib import Path

    root = Path(credentials.__file__).parent.parent
    for path in root.rglob("*.py"):
        text = path.read_text()
        for token in ("load_dotenv", "dotenv_values", "from dotenv"):
            assert token not in text


# --- the LiteLLM transport ---------------------------------------------------------

def test_the_request_names_the_qualified_model():
    completion = FakeCompletion()
    provider_for("anthropic", completion=completion).complete_structured(
        AIRequest(system="s", user="u"))
    assert completion.calls[0]["model"] == "anthropic/claude-sonnet-4-5"


def test_the_credential_is_passed_explicitly_not_through_the_environment(
        monkeypatch):
    """Exporting a key would expose it to every child of this process."""
    import os

    before = dict(os.environ)
    completion = FakeCompletion()
    provider_for("openai", completion=completion).complete_structured(
        AIRequest(system="s", user="u"))

    assert completion.calls[0]["api_key"] == SENTINEL
    assert os.environ == before, "the provider mutated the environment"


def test_a_local_provider_sends_no_api_key():
    completion = FakeCompletion()
    LiteLLMProvider(
        configuration=provider_config.build_configuration("ollama"),
        api_key=None, completion=completion,
    ).complete_structured(AIRequest(system="s", user="u"))
    assert "api_key" not in completion.calls[0]


def test_the_request_grants_no_tools():
    completion = FakeCompletion()
    provider_for(completion=completion).complete_structured(
        AIRequest(system="s", user="u"))
    call = completion.calls[0]
    assert call["tools"] is None
    assert call["stream"] is False
    for forbidden in ("functions", "tool_choice", "callbacks", "mcp_servers"):
        assert forbidden not in call


def test_structured_output_is_requested_but_not_relied_upon():
    completion = FakeCompletion()
    provider_for(completion=completion).complete_structured(
        AIRequest(system="s", user="u"))
    assert completion.calls[0]["response_format"] == {"type": "json_object"}


def test_the_payload_is_exactly_the_two_sanitized_messages():
    completion = FakeCompletion()
    provider_for(completion=completion).complete_structured(
        AIRequest(system="SYSTEM TEXT", user="USER TEXT"))
    messages = completion.calls[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == "SYSTEM TEXT"
    assert messages[1]["content"] == "USER TEXT"


def test_the_key_never_appears_in_the_prompt_or_the_repr():
    completion = FakeCompletion()
    provider = provider_for(completion=completion)
    provider.complete_structured(AIRequest(system="s", user="u"))

    payload = json.dumps(completion.calls[0]["messages"])
    assert SENTINEL not in payload
    assert SENTINEL not in repr(provider)
    assert SENTINEL not in str(provider)


def test_the_key_never_appears_in_a_provider_error():
    completion = FakeCompletion(
        error=RuntimeError(f"request failed with key={SENTINEL}"))
    with pytest.raises(AIError) as caught:
        provider_for(completion=completion).complete_structured(
            AIRequest(system="s", user="u"))
    assert SENTINEL not in str(caught.value)


# --- provider failures become DelegateDoctor errors ---------------------------------

@pytest.mark.parametrize("message, expected", [
    ("AuthenticationError: invalid api key", "AUTHENTICATION FAILED"),
    ("RateLimitError: rate limit exceeded", "RATE LIMIT"),
    ("insufficient quota for this billing account", "QUOTA OR BILLING"),
    ("maximum context length exceeded for model", "CONTEXT LIMIT"),
    ("Connection refused; service unavailable", "UNAVAILABLE"),
])
def test_provider_failures_are_translated(message, expected):
    completion = FakeCompletion(error=RuntimeError(message))
    with pytest.raises(AIError) as caught:
        provider_for(completion=completion).complete_structured(
            AIRequest(system="s", user="u"))
    assert expected in str(caught.value)


def test_a_provider_failure_does_not_expose_a_stack_trace():
    completion = FakeCompletion(error=RuntimeError("x" * 5000))
    with pytest.raises(AIError) as caught:
        provider_for(completion=completion).complete_structured(
            AIRequest(system="s", user="u"))
    assert len(str(caught.value)) < 1000


def test_an_empty_reply_is_an_error():
    with pytest.raises(AIError):
        provider_for(completion=FakeCompletion(text="")).complete_structured(
            AIRequest(system="s", user="u"))


def test_an_unexpected_response_shape_is_an_empty_response():
    """A response with no choices produced nothing. It is not a candidate."""
    from delegate_doctor.agent import provider_response

    class Broken:
        def __call__(self, **kwargs):
            return object()

    result = provider_for(completion=Broken()).complete(
        AIRequest(system="s", user="u"))
    assert result.status == provider_response.EMPTY
    assert result.reported_status == "PROVIDER_EMPTY_RESPONSE"

    # And callers that want an exception still get one, with the outcome named.
    with pytest.raises(AIError) as caught:
        provider_for(completion=Broken()).complete_structured(
            AIRequest(system="s", user="u"))
    assert "PROVIDER_EMPTY_RESPONSE" in str(caught.value)


# --- LiteLLM must not expand the trust surface -------------------------------------

def test_litellm_privacy_settings_are_applied():
    class FakeLiteLLM:
        pass

    module = FakeLiteLLM()
    client._configure_litellm_privacy(module)

    assert module.set_verbose is False
    assert module.turn_off_message_logging is True
    assert module.telemetry is False
    for callback in ("success_callback", "failure_callback", "callbacks",
                     "input_callback"):
        assert getattr(module, callback) == []


def test_no_observability_integration_is_enabled():
    source = open(client.__file__).read()
    for integration in ("langfuse", "helicone", "datadog", "opentelemetry",
                        "otel", "lunary", "prometheus"):
        assert integration not in source.lower()


def test_the_transport_does_not_enable_tool_or_agent_features():
    source = open(client.__file__).read()
    for feature in ("mcp", "function_call", "tool_choice", "add_function_to_prompt"):
        assert feature not in source.lower()


def test_the_pinned_litellm_version_is_recorded():
    assert client.LITELLM_VERSION == "1.96.2"


def test_litellm_is_pinned_in_pyproject():
    from pathlib import Path

    text = (Path(client.__file__).parent.parent.parent / "pyproject.toml").read_text()
    assert 'litellm==1.96.2' in text


# --- --no-ai --------------------------------------------------------------------

def test_refusing_ai_prevents_a_provider_from_existing():
    from delegate_doctor.agent.client import AIDisabled

    with pytest.raises(AIDisabled) as caught:
        REAL_BUILD_PROVIDER(allow_ai=False)
    assert "not permitted" in str(caught.value)


def test_refusing_ai_resolves_no_credential(monkeypatch):
    """Nothing is even read: the refusal is structural, not a policy check."""
    monkeypatch.setattr(credentials, "resolve_api_key",
                        lambda provider="": pytest.fail("a credential was read"))
    with pytest.raises(client.AIDisabled):
        REAL_BUILD_PROVIDER(allow_ai=False)
