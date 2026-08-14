"""The boundary between this machine and anything else.

Two directions matter, and they are different problems:

  * **Outbound.** Model source is potentially private intellectual property. If
    any of it is ever sent to an AI provider, it is scrubbed first and the user
    is told exactly what will leave. Weights, tensor values, environment
    variables and unrelated files are never candidates at all.

  * **Downward.** DelegateDoctor runs the user's model in a child process. That
    child must not inherit an AI API key, a cloud credential or a source-control
    token, because a model file is ordinary Python and can read its own
    environment.

Both directions are enforced by construction rather than by care: outbound text
is built from an allowlist and then redacted, and the child environment is built
by *removing* known-dangerous names rather than by hoping none are set.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

PLACEHOLDER = "[REDACTED]"

# --- environment variables a child process must never inherit ---------------
#
# Exact names first. A model file is ordinary Python: it can read os.environ,
# and nothing about running it should hand it the user's cloud account.

SECRET_ENVIRONMENT_NAMES = frozenset({
    # DelegateDoctor's own
    "DELEGATE_DOCTOR_LLM_API_KEY",
    # LLM providers
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "MISTRAL_API_KEY", "COHERE_API_KEY", "GROQ_API_KEY", "TOGETHER_API_KEY",
    "PERPLEXITY_API_KEY", "XAI_API_KEY", "DEEPSEEK_API_KEY",
    "AZURE_OPENAI_API_KEY", "OPENAI_ORGANIZATION",
    # cloud
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN", "GOOGLE_APPLICATION_CREDENTIALS",
    "AZURE_CLIENT_SECRET", "AZURE_TENANT_ID", "AZURE_CLIENT_ID",
    # source control and model hubs
    "GITHUB_TOKEN", "GH_TOKEN", "GITLAB_TOKEN", "GIT_ASKPASS",
    "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN",
    "WANDB_API_KEY", "COMET_API_KEY", "NEPTUNE_API_TOKEN",
    # package registries and services
    "NPM_TOKEN", "PYPI_TOKEN", "TWINE_PASSWORD", "DOCKER_PASSWORD",
    "SLACK_TOKEN", "STRIPE_SECRET_KEY", "SENTRY_AUTH_TOKEN",
    "NETRC", "SSH_AUTH_SOCK",
})

# Shape-based backstop, for the names nobody thought of. A variable called
# ACME_INTERNAL_API_KEY is a secret whether or not it is on the list above.
SECRET_NAME_PATTERN = re.compile(
    r"(_?API_?KEY|_TOKEN|TOKEN$|_SECRET|SECRET$|_PASSWORD|PASSWORD$|"
    r"_CREDENTIALS|_PRIVATE_KEY|_ACCESS_KEY|_AUTH$|PASSWD)",
    re.IGNORECASE,
)

# Values shorter than this are ignored when scrubbing text: replacing a
# 3-character value would mangle unrelated prose without protecting anything.
MINIMUM_SECRET_LENGTH = 8

# --- credential shapes found in text ----------------------------------------

INLINE_SECRET_PATTERNS = (
    # provider key formats
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}"),
    # headers and assignments
    re.compile(r"(?i)\bauthorization\s*[:=]\s*\S+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"(?i)\b(api[_-]?key|secret|password|passwd|token|access[_-]?key"
               r"|key|auth)\s*[:=]\s*['\"]?[A-Za-z0-9._\-/+]{8,}['\"]?"),
    # credential URLs: https://user:password@host
    re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://[^\s/@:]+:[^\s/@]+@"),
)

PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)


def secret_environment_values() -> list:
    """Values of the current process's secret-looking variables.

    Collected only so they can be searched for in outgoing text. Never stored,
    never logged, never sent.
    """
    values = []
    for name, value in os.environ.items():
        if not value or len(value) < MINIMUM_SECRET_LENGTH:
            continue
        if name in SECRET_ENVIRONMENT_NAMES or SECRET_NAME_PATTERN.search(name):
            values.append(value)
    # Longest first, so a secret containing another is replaced whole.
    return sorted(set(values), key=len, reverse=True)


def redact(text) -> str:
    """Scrub credential material out of a string.

    Applied to anything DelegateDoctor prints, writes to an artifact, or would
    send to a provider. Deliberately aggressive about credentials and silent
    about everything else.
    """
    if text is None:
        return ""
    text = str(text)

    # 1. exact values known to be secret in this process
    for value in secret_environment_values():
        if value in text:
            text = text.replace(value, PLACEHOLDER)

    # 2. whole private-key blocks, before the line-level patterns see them
    text = PRIVATE_KEY_BLOCK.sub(PLACEHOLDER, text)

    # 3. recognisable credential shapes
    for pattern in INLINE_SECRET_PATTERNS:
        text = pattern.sub(PLACEHOLDER, text)

    return text


def redact_home_paths(text: str, home: str = None) -> str:
    """Replace the user's home directory with `~`.

    An absolute path leaks a username, and often an employer and a project
    layout, none of which any provider needs to answer a question about a
    graph.
    """
    if text is None:
        return ""
    text = str(text)
    home_path = str(home if home is not None else Path.home())
    if home_path and home_path != os.sep:
        text = text.replace(home_path, "~")
    return text


def sanitize_for_transmission(text: str, home: str = None) -> str:
    """Everything applied before a single character leaves this machine."""
    return redact_home_paths(redact(text), home=home)


def contains_secret(text: str) -> bool:
    """Does this text still look like it carries a credential?

    Used as a stop-check: if redaction could not clean something, the caller
    asks the user rather than sending it.
    """
    if not text:
        return False
    if PRIVATE_KEY_BLOCK.search(text):
        return True
    return any(pattern.search(text) for pattern in INLINE_SECRET_PATTERNS)


# --- the child process environment ------------------------------------------


def sanitized_child_environment(base: dict = None, extra_removals=()) -> dict:
    """An environment safe to hand to a user model or a generated adapter.

    Built by removal, so a variable nobody anticipated is still dropped if it
    *looks* like a credential. The child keeps everything it needs to import
    torch and run - PATH, HOME, VIRTUAL_ENV, PYTHONPATH and the rest.
    """
    source = os.environ if base is None else base
    removals = set(extra_removals) | SECRET_ENVIRONMENT_NAMES

    child = {}
    for name, value in source.items():
        if name in removals:
            continue
        if SECRET_NAME_PATTERN.search(name):
            continue
        child[name] = value
    return child


def assert_no_secrets(text: str, label: str = "output") -> None:
    """Raise if a credential would escape. For belt-and-braces call sites."""
    if contains_secret(text):
        raise RuntimeError(
            f"DelegateDoctor refused to emit {label}: it still contains "
            f"credential-shaped material after redaction."
        )


# --- what the user is told before source leaves the machine -----------------


def consent_disclosure(files) -> str:
    """The exact, specific disclosure shown before the first transmission."""
    listed = "\n".join(f"  {Path(path).name}" for path in files)
    return (
        "AI preparation needs to send selected source code to the configured\n"
        "provider.\n"
        "\n"
        "Files:\n"
        f"{listed}\n"
        "\n"
        "Model weights, checkpoint contents, input tensors, environment\n"
        "variables, credentials and unrelated project files will NOT be sent.\n"
    )
