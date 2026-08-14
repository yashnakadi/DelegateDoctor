"""Where the AI API key comes from: your environment, and nowhere else.

    DelegateDoctor does not store your AI API key.

There is no credential store, no keychain entry, no config file holding a
secret, no `.env` read or written, and no shell profile edited. The key is read
from the current process environment at the moment a request is made, handed to
LiteLLM as an argument, and dropped.

That makes the security claim short enough to verify by reading this file:
DelegateDoctor cannot leak a stored key, because it never stores one.

Resolution order
----------------
    1. DELEGATE_DOCTOR_LLM_API_KEY   the generic variable
    2. the provider's own conventional variable (ANTHROPIC_API_KEY, ...)
    3. nothing, and AI is unavailable

The generic variable supplies a *credential only*. It never selects the
provider, the model or the endpoint - those come from the non-secret
configuration in `provider_config.py`. If the wrong provider's key is exported,
the provider rejects it, and that is a clearer failure than DelegateDoctor
guessing from a key prefix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

ENVIRONMENT_VARIABLE = "DELEGATE_DOCTOR_LLM_API_KEY"

SOURCE_GENERIC = f"{ENVIRONMENT_VARIABLE} environment variable"
SOURCE_NONE = "not configured"
SOURCE_NOT_REQUIRED = "Not required"


@dataclass(frozen=True)
class KeyStatus:
    """Whether a credential is available, and from where. Never the key itself.

    Only availability and the *name* of the source are recorded. No prefix, no
    suffix, no length, no masked form: a partially revealed key is still a
    disclosure, and there is no question it usefully answers.
    """

    available: bool
    source: str
    required: bool = True

    def describe(self) -> str:
        if not self.required:
            return f"API key                  {SOURCE_NOT_REQUIRED}"
        return f"API key                  {self.source}"


def _provider_variable(provider: str) -> str:
    """The provider's own conventional variable name, if it has one."""
    from . import provider_config

    definition = provider_config.PROVIDERS_BY_KEY.get(provider)
    return definition.environment_variable if definition else ""


def resolve_api_key(provider: str = "") -> str | None:
    """The user's key for this provider, from the environment, or None.

    Callers must not log, store or print the result, and should drop their
    reference as soon as the request has been made.
    """
    generic = os.environ.get(ENVIRONMENT_VARIABLE)
    if generic:
        return generic

    variable = _provider_variable(provider)
    if variable:
        return os.environ.get(variable) or None
    return None


def key_status(configuration=None) -> KeyStatus:
    """Whether AI can authenticate, for display. Never returns the key.

    Provider-aware: a local provider that needs no credential is reported as
    "Not required" rather than as unavailable, which would be misleading.
    """
    provider = getattr(configuration, "provider", "") if configuration else ""

    if configuration is not None and not configuration.needs_api_key:
        return KeyStatus(available=True, source=SOURCE_NOT_REQUIRED,
                         required=False)

    if os.environ.get(ENVIRONMENT_VARIABLE):
        return KeyStatus(True, SOURCE_GENERIC)

    variable = _provider_variable(provider)
    if variable and os.environ.get(variable):
        return KeyStatus(True, f"{variable} environment variable")

    return KeyStatus(False, SOURCE_NONE)


# --- messages ----------------------------------------------------------------


NOT_CONFIGURED_MESSAGE = (
    "AI NOT CONFIGURED\n"
    "\n"
    "No AI provider has been chosen yet.\n"
    "\n"
    "    delegate-doctor configure-ai\n"
    "\n"
    "or skip AI entirely by analyzing the model in Python:\n"
    "\n"
    "    import torch\n"
    "    from delegate_doctor import optimize\n"
    "\n"
    "    result = optimize(model.eval(), args=(example_input,))\n"
    "\n"
    "The Python API never uses AI and never needs a key."
)


def missing_key_message(configuration) -> str:
    """Provider and model *are* configured; only the credential is absent.

    Deliberately does not suggest `configure-ai`: re-running it would change
    nothing, because it never asks for a key.
    """
    definition = configuration.definition
    variable = definition.environment_variable

    alternative = ""
    if variable and variable != ENVIRONMENT_VARIABLE:
        alternative = f"\nor:\n\n    export {variable}=\"...\"\n"

    return (
        f"AI NOT CONFIGURED\n"
        f"\n"
        f"No credential is available for {definition.label}.\n"
        f"\n"
        f"DelegateDoctor does not store API keys.\n"
        f"\n"
        f"Set one in this terminal:\n"
        f"\n"
        f"    export {ENVIRONMENT_VARIABLE}=\"...\"\n"
        f"{alternative}"
        f"\nThen run DelegateDoctor again.\n"
        f"\n"
        f"To skip AI entirely, analyze the model in Python:\n"
        f"\n"
        f"    from delegate_doctor import optimize\n"
        f"    result = optimize(model.eval(), args=(example_input,))"
    )


def environment_instructions(configuration) -> str:
    """The copyable lines printed after `configure-ai`."""
    definition = configuration.definition
    if not configuration.needs_api_key:
        text = "No API key required for this local provider."
        if definition.note:
            text += f"\n{definition.note}"
        return text

    variable = definition.environment_variable
    text = (f"Set your API key for this terminal:\n"
            f"\n"
            f"    export {ENVIRONMENT_VARIABLE}=\"...\"\n")
    if variable and variable != ENVIRONMENT_VARIABLE:
        text += (f"\nAlternatively:\n"
                 f"\n"
                 f"    export {variable}=\"...\"\n")
    text += ("\nDelegateDoctor does not store API keys. It reads the key from\n"
             "your environment only when an AI request is needed.")
    return text


# --- configure-ai --------------------------------------------------------------


def configure_interactively(prompt=input, announce=print) -> int:
    """`delegate-doctor configure-ai` - choose a provider and model.

    Configuration and authentication are separate concerns, so this asks for
    neither a key nor permission to store one. It succeeds with no credential
    present anywhere, makes no provider request, and writes only non-secret
    values.
    """
    from . import provider_config

    announce("DelegateDoctor AI Setup\n")
    announce("AI is optional. The Python API workflow - optimize() on a live")
    announce("model - never uses it.\n")
    announce("DelegateDoctor uses your own AI provider account. Requests may")
    announce("incur charges from that provider.\n")
    announce(provider_config.describe_menu())

    try:
        answer = prompt("\nSelect provider [2]: ").strip()
    except (EOFError, KeyboardInterrupt):
        announce("\nCancelled. Nothing was saved.")
        return 1

    definitions = provider_config.PROVIDERS
    if not answer:
        definition = provider_config.PROVIDERS_BY_KEY["anthropic"]
    elif answer.isdigit() and 1 <= int(answer) <= len(definitions):
        definition = definitions[int(answer) - 1]
    else:
        announce(f"\nNot a valid choice: {answer!r}")
        return 2

    try:
        model_answer = prompt(
            f"\nModel [{definition.default_model or 'required'}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        announce("\nCancelled. Nothing was saved.")
        return 1

    if definition.key == "advanced" and not model_answer:
        announce("\nAn advanced configuration needs an explicit LiteLLM model "
                 "string, for example 'azure/my-deployment'.")
        return 2

    api_base = ""
    if definition.is_local:
        try:
            api_base = prompt("\nAPI base [default local endpoint]: ").strip()
        except (EOFError, KeyboardInterrupt):
            api_base = ""

    try:
        configuration = provider_config.build_configuration(
            definition.key, model_answer, api_base)
        saved_to = provider_config.save_configuration(configuration)
    except provider_config.ProviderConfigError as error:
        announce(f"\n{error}")
        return 2

    announce("\nAI configured successfully.\n")
    announce(f"Provider                 {definition.label}")
    announce(f"Model                    {configuration.model}")
    announce(f"Source transmission      {configuration.processing_label}")
    announce(f"\nConfiguration:\n  {saved_to}")
    announce(f"\n{environment_instructions(configuration)}")
    return 0
