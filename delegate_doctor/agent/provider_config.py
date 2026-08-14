"""Which provider and model to use. Never the credential.

DelegateDoctor is bring-your-own-key: it has no account, ships no key, and
funds no inference. This module holds the *non-secret* half of that - which
provider the user picked and which model - and keeps it strictly separate from
the credential, which DelegateDoctor never stores at all and reads from the
environment (`credentials.py`).

The split is the point. Provider and model are ordinary settings and are stored
in a small JSON file; a test asserts that file can never contain a key.

Local versus remote is decided here too, because it changes what DelegateDoctor
has to ask permission for. A local Ollama never sees the network, so warning
about "source leaving your machine" would be both false and annoying - but a
custom base URL is *not* assumed to be local just because it is unusual.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIRECTORY_NAME = ".delegate-doctor"
CONFIG_FILE_NAME = "ai.json"

# Where a model string may not go: control characters would let a value smuggle
# a header break, and a credential URL would put a secret in a non-secret file.
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
_CREDENTIAL_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://[^\s/@]+:[^\s/@]+@")
MAX_MODEL_LENGTH = 200

PROCESSING_LOCAL = "LOCAL ONLY"
PROCESSING_REMOTE = "REMOTE PROVIDER"


class ProviderConfigError(RuntimeError):
    """The provider or model configuration is not usable."""


@dataclass(frozen=True)
class ProviderDefinition:
    """One first-class provider, and how LiteLLM expects to be told about it."""

    key: str
    label: str
    model_prefix: str
    default_model: str
    # The environment variable LiteLLM/this provider conventionally reads. Used
    # only as a documented fallback; the credential store is preferred.
    environment_variable: str = ""
    needs_api_key: bool = True
    is_local: bool = False
    note: str = ""

    def qualify(self, model: str) -> str:
        """LiteLLM's canonical `provider/model` form."""
        if not self.model_prefix:
            return model
        if model.startswith(f"{self.model_prefix}/"):
            return model
        return f"{self.model_prefix}/{model}"


# The onboarding menu. Short on purpose: LiteLLM supports a hundred providers,
# and putting them all here would make the common case worse. Anything else is
# reachable through the advanced entry.
PROVIDERS = (
    ProviderDefinition(
        key="openai", label="OpenAI", model_prefix="openai",
        default_model="gpt-4o", environment_variable="OPENAI_API_KEY"),
    ProviderDefinition(
        key="anthropic", label="Anthropic", model_prefix="anthropic",
        default_model="claude-sonnet-4-5",
        environment_variable="ANTHROPIC_API_KEY"),
    ProviderDefinition(
        key="gemini", label="Google Gemini", model_prefix="gemini",
        default_model="gemini-2.0-flash",
        environment_variable="GEMINI_API_KEY"),
    ProviderDefinition(
        key="openrouter", label="OpenRouter", model_prefix="openrouter",
        default_model="anthropic/claude-sonnet-4-5",
        environment_variable="OPENROUTER_API_KEY"),
    ProviderDefinition(
        key="ollama", label="Ollama / local", model_prefix="ollama_chat",
        default_model="qwen2.5-coder:7b", needs_api_key=False, is_local=True,
        note="Runs on this machine. Nothing is sent to a remote provider."),
    ProviderDefinition(
        key="advanced", label="Advanced LiteLLM provider", model_prefix="",
        default_model="", environment_variable="DELEGATE_DOCTOR_LLM_API_KEY",
        note="Any model string LiteLLM understands, e.g. 'azure/my-deployment'."),
)

PROVIDERS_BY_KEY = {definition.key: definition for definition in PROVIDERS}


@dataclass(frozen=True)
class AIConfiguration:
    """The non-secret configuration. Deliberately holds no credential."""

    provider: str
    model: str
    api_base: str = ""

    @property
    def definition(self) -> ProviderDefinition:
        definition = PROVIDERS_BY_KEY.get(self.provider)
        if definition is None:
            raise ProviderConfigError(f"Unknown provider: {self.provider!r}")
        return definition

    @property
    def litellm_model(self) -> str:
        return self.definition.qualify(self.model)

    @property
    def is_local(self) -> bool:
        """Local only when the provider is local *and* nothing redirects it.

        A custom `api_base` is never assumed to be local: "localhost" in a
        hostname is not proof, and being wrong here would mean telling a user
        their source stayed on the machine when it did not.
        """
        if not self.definition.is_local:
            return False
        return not self.api_base or _is_loopback(self.api_base)

    @property
    def processing_label(self) -> str:
        return PROCESSING_LOCAL if self.is_local else PROCESSING_REMOTE

    @property
    def needs_api_key(self) -> bool:
        return self.definition.needs_api_key

    def describe(self) -> str:
        return f"{self.definition.label} · {self.model}"

    def to_dict(self) -> dict:
        payload = {"provider": self.provider, "model": self.model}
        if self.api_base:
            payload["api_base"] = self.api_base
        return payload


def _is_loopback(api_base: str) -> bool:
    """Only an unambiguous loopback address counts as local."""
    from urllib.parse import urlparse

    try:
        host = (urlparse(api_base).hostname or "").lower()
    except Exception:
        return False
    return host in ("127.0.0.1", "localhost", "::1", "0.0.0.0"[:0] or "localhost")


# --- validation ---------------------------------------------------------------


def validate_model(model: str) -> str:
    """A model string must be plain, bounded text with no injection surface."""
    if not isinstance(model, str) or not model.strip():
        raise ProviderConfigError("The model name cannot be empty.")
    model = model.strip()
    if len(model) > MAX_MODEL_LENGTH:
        raise ProviderConfigError(
            f"The model name is too long (>{MAX_MODEL_LENGTH} characters).")
    if _CONTROL_CHARACTERS.search(model):
        raise ProviderConfigError(
            "The model name contains control characters.")
    if _CREDENTIAL_URL.search(model):
        raise ProviderConfigError(
            "The model name looks like it contains a credential. Model names "
            "are stored in a non-secret file, so DelegateDoctor will not "
            "accept one.")
    return model


def validate_api_base(api_base: str) -> str:
    if not api_base:
        return ""
    api_base = api_base.strip()
    if _CONTROL_CHARACTERS.search(api_base) or _CREDENTIAL_URL.search(api_base):
        raise ProviderConfigError(
            "The API base URL is not accepted: it contains control characters "
            "or embedded credentials.")
    if len(api_base) > MAX_MODEL_LENGTH:
        raise ProviderConfigError("The API base URL is too long.")
    return api_base


def build_configuration(provider: str, model: str = "",
                        api_base: str = "") -> AIConfiguration:
    definition = PROVIDERS_BY_KEY.get(provider)
    if definition is None:
        raise ProviderConfigError(
            f"Unknown provider {provider!r}. Choose one of: "
            f"{', '.join(item.key for item in PROVIDERS)}")
    chosen = validate_model(model or definition.default_model)
    return AIConfiguration(provider=provider, model=chosen,
                           api_base=validate_api_base(api_base))


# --- persistence (non-secret only) --------------------------------------------


def project_root(start: Path | None = None) -> Path:
    """Find the DelegateDoctor repository root from the current directory."""
    current = (start or Path.cwd()).resolve()

    for candidate in (current, *current.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "delegate_doctor").is_dir()
        ):
            return candidate

    raise ProviderConfigError(
        "Could not find the DelegateDoctor project root. "
        "Run this command from inside the DelegateDoctor repository."
    )


def config_directory() -> Path:
    """Project-local DelegateDoctor configuration directory."""
    return project_root() / CONFIG_DIRECTORY_NAME


def config_path() -> Path:
    return config_directory() / CONFIG_FILE_NAME


# Keys that must never be written here, whatever a caller passes.
_FORBIDDEN_CONFIG_FIELDS = ("api_key", "key", "secret", "token", "password",
                            "credential", "authorization")


def save_configuration(configuration: AIConfiguration, path: Path = None) -> Path:
    """Write provider and model. Refuses to write anything secret-shaped."""
    payload = configuration.to_dict()
    for field in payload:
        if any(forbidden in field.lower() for forbidden in _FORBIDDEN_CONFIG_FIELDS):
            raise ProviderConfigError(
                f"Refusing to write {field!r} to the configuration file. "
                f"DelegateDoctor does not store credentials; supply the key "
                f"through the environment instead.")

    target = Path(path) if path else config_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        # A read-only or unwritable config directory is an ordinary environment
        # problem, not a crash. No credential is involved - configure-ai never
        # handles one - so a failure here cannot leave a partial secret.
        raise ProviderConfigError(
            f"Could not write DelegateDoctor AI configuration:\n"
            f"\n"
            f"  {target}\n"
            f"\n"
            f"{type(error).__name__}: check that the parent directory exists "
            f"and is writable."
        )
    return target


def load_configuration(path: Path = None) -> AIConfiguration | None:
    """The saved configuration, or None. Never raises.

    "Run from outside the repository" is not a different answer from "nothing
    configured": in both cases there is no configuration to load, and the
    caller's job is to say AI is not set up - not to emit a traceback. Only
    `save_configuration` genuinely needs a project root, and it still says so.
    """
    try:
        target = Path(path) if path else config_path()
    except ProviderConfigError:
        return None
    if not target.is_file():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return build_configuration(
            provider=str(payload.get("provider", "")),
            model=str(payload.get("model", "")),
            api_base=str(payload.get("api_base", "")),
        )
    except ProviderConfigError:
        return None


def describe_menu() -> str:
    lines = ["Provider:", ""]
    for position, definition in enumerate(PROVIDERS, start=1):
        lines.append(f"  {position}. {definition.label}")
    return "\n".join(lines)
