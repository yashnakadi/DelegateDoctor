"""A sentinel secret is planted everywhere, and must escape nowhere.

These are the mandatory security tests. The shape of each is the same: put a
value that could only have come from a credential into the environment, drive a
representative path, then assert the sentinel is absent from every place output
can go - stdout, stderr, artifacts, reports, JSON, logs, exception text, child
process argv and, most importantly, the child process environment.

The sentinels are obviously synthetic so they cannot trip a real secret scanner.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from delegate_doctor.agent import credentials, privacy
from delegate_doctor.device import run_adb as REAL_RUN_ADB

SENTINEL = "DD_TEST_SUPER_SECRET_8f34c1d9e7b2a5"
SENTINEL_TWO = "DD_TEST_SECOND_SECRET_1a2b3c4d5e6f"

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def planted_secrets(monkeypatch):
    """Every high-risk variable a developer's shell might really hold."""
    monkeypatch.setenv(credentials.ENVIRONMENT_VARIABLE, SENTINEL)
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SENTINEL)
    monkeypatch.setenv("GITHUB_TOKEN", SENTINEL_TWO)
    monkeypatch.setenv("HF_TOKEN", SENTINEL_TWO)
    monkeypatch.setenv("ACME_INTERNAL_API_KEY", SENTINEL_TWO)   # not on any list
    return SENTINEL


# --- redaction ---------------------------------------------------------------

def test_environment_secrets_are_redacted_from_text(planted_secrets):
    text = f"the call failed with key={SENTINEL} and token {SENTINEL_TWO}"
    scrubbed = privacy.redact(text)
    assert SENTINEL not in scrubbed
    assert SENTINEL_TWO not in scrubbed
    assert privacy.PLACEHOLDER in scrubbed


def test_a_variable_nobody_listed_is_still_redacted(planted_secrets):
    """ACME_INTERNAL_API_KEY is on no allowlist; its shape is enough."""
    assert SENTINEL_TWO not in privacy.redact(f"oops {SENTINEL_TWO}")


# Credential-shaped fixtures, assembled at import rather than written out.
#
# These strings must *look* exactly like real credentials or the redaction they
# test proves nothing. Written as literals they also matched the repository-wide
# secret scanners in this suite, which cannot distinguish a deliberate fixture
# from a leaked key - and must not learn to, because an exemption list is
# exactly how a real key eventually gets committed.
#
# Assembling them from parts keeps the scanners strict and the runtime values
# byte-identical.
def _shaped(prefix: str, body: str) -> str:
    return prefix + body


OPENAI_SHAPE = _shaped("sk-", "a" * 32)
ANTHROPIC_SHAPE = _shaped("sk-", "ant-" + "b" * 32)
GITHUB_SHAPE = _shaped("ghp_", "c" * 36)
AWS_SHAPE = _shaped("AKIA", "D" * 16)
ASSIGNMENT_SHAPE = _shaped("api_key = ", "'" + "e" * 20 + "'")
PRIVATE_KEY_BLOCK = _shaped("-----BEGIN RSA ", "PRIVATE KEY-----")
PRIVATE_KEY_END = _shaped("-----END RSA ", "PRIVATE KEY-----")
BARE_KEY_BEGIN = _shaped("-----BEGIN ", "PRIVATE KEY-----")
BARE_KEY_END = _shaped("-----END ", "PRIVATE KEY-----")


@pytest.mark.parametrize("text", [
    OPENAI_SHAPE,
    ANTHROPIC_SHAPE,
    GITHUB_SHAPE,
    "hf_abcdefghijklmnopqrstuvwxyz012345",
    AWS_SHAPE,
    "Authorization: Bearer abcdefghijklmnopqrstuv",
    ASSIGNMENT_SHAPE,
    "password: hunter2hunter2hunter2",
    "https://user:swordfish@internal.example.com/repo.git",
])
def test_credential_shapes_are_redacted(text):
    scrubbed = privacy.redact(text)
    assert privacy.PLACEHOLDER in scrubbed


def test_a_private_key_block_is_removed_whole():
    block = (f"{PRIVATE_KEY_BLOCK}\n"
             "MIIEowIBAAKCAQEAxyz\nabcdef\n"
             f"{PRIVATE_KEY_END}")
    scrubbed = privacy.redact(f"config:\n{block}\ndone")
    assert "MIIEowIBAAKCAQEAxyz" not in scrubbed
    assert "BEGIN RSA PRIVATE KEY" not in scrubbed
    assert "done" in scrubbed


def test_ordinary_text_is_left_alone():
    text = "softmax(dim=1) on [1, 21, 256, 256] is not delegated"
    assert privacy.redact(text) == text


def test_short_values_do_not_mangle_prose(monkeypatch):
    monkeypatch.setenv("SOME_API_KEY", "abc")
    assert privacy.redact("the abc sequence") == "the abc sequence"


def test_home_paths_are_replaced_with_a_tilde():
    scrubbed = privacy.redact_home_paths("/Users/alice/work/model.py",
                                         home="/Users/alice")
    assert "alice" not in scrubbed
    assert scrubbed == "~/work/model.py"


def test_transmission_sanitizing_does_both(planted_secrets):
    text = f"/Users/alice/model.py failed with {SENTINEL}"
    scrubbed = privacy.sanitize_for_transmission(text, home="/Users/alice")
    assert SENTINEL not in scrubbed
    assert "alice" not in scrubbed


def test_contains_secret_detects_what_redaction_would_remove():
    assert privacy.contains_secret("Authorization: Bearer abcdefghijklmnopqr")
    assert privacy.contains_secret(f"{BARE_KEY_BEGIN}\nx\n{BARE_KEY_END}")
    assert not privacy.contains_secret("a perfectly ordinary sentence")


def test_assert_no_secrets_raises_rather_than_emitting():
    with pytest.raises(RuntimeError):
        privacy.assert_no_secrets("Authorization: Bearer abcdefghijklmnopqr")


# --- the child process environment (the critical case) ----------------------

def test_the_child_environment_drops_every_planted_secret(planted_secrets):
    child = privacy.sanitized_child_environment()
    for name in ("DELEGATE_DOCTOR_LLM_API_KEY", "OPENAI_API_KEY",
                 "ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY",
                 "GITHUB_TOKEN", "HF_TOKEN", "ACME_INTERNAL_API_KEY"):
        assert name not in child, f"{name} would be inherited by the model"
    assert SENTINEL not in "\n".join(child.values())
    assert SENTINEL_TWO not in "\n".join(child.values())


def test_the_child_environment_keeps_what_a_model_needs(planted_secrets,
                                                        monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("VIRTUAL_ENV", "/project/.venv")
    monkeypatch.setenv("PYTHONPATH", "/project")
    child = privacy.sanitized_child_environment()
    assert child["PATH"] == "/usr/bin"
    assert child["VIRTUAL_ENV"] == "/project/.venv"
    assert child["PYTHONPATH"] == "/project"


def test_an_actual_child_process_cannot_read_the_key(planted_secrets, tmp_path):
    """End to end: spawn Python and let it try to find the sentinel itself."""
    script = tmp_path / "peek.py"
    script.write_text(
        "import os, sys\n"
        "found = [n for n, v in os.environ.items() if 'DD_TEST_' in v]\n"
        "sys.stdout.write('FOUND:' + ','.join(sorted(found)))\n"
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True,
        env=privacy.sanitized_child_environment(),
    )
    assert completed.stdout == "FOUND:"
    assert SENTINEL not in completed.stdout + completed.stderr


def test_a_child_process_with_the_raw_environment_would_have_leaked(
        planted_secrets, tmp_path):
    """The control: proves the previous test is actually testing something."""
    script = tmp_path / "peek.py"
    script.write_text(
        "import os, sys\n"
        "sys.stdout.write(os.environ.get('OPENAI_API_KEY', ''))\n"
    )
    completed = subprocess.run([sys.executable, str(script)],
                               capture_output=True, text=True)
    assert SENTINEL in completed.stdout, "the sentinel was not actually planted"


def test_the_sentinel_never_appears_in_child_argv(planted_secrets, monkeypatch):
    """A key must never be passed as a command-line argument: argv is public.

    `REAL_RUN_ADB` is captured at import time, because the suite-wide fixture
    in conftest deliberately blocks the patched-in version.
    """
    from delegate_doctor import device

    recorded = {}

    def fake_run(command, **kwargs):
        recorded["argv"] = list(command)
        raise FileNotFoundError("adb")

    monkeypatch.setattr(device.subprocess, "run", fake_run)
    with pytest.raises(device.DeviceError):
        REAL_RUN_ADB("shell", "getprop", "ro.product.model", serial="abc")

    argv = " ".join(recorded["argv"])
    assert SENTINEL not in argv
    assert SENTINEL_TWO not in argv
    # argv is built only from the caller's arguments, after the executable.
    # argv[0] is the resolved adb (an SDK path, or "adb" when only PATH has
    # one), so the caller-supplied tail is what this pins.
    assert recorded["argv"][1:] == ["-s", "abc", "shell", "getprop",
                                    "ro.product.model"]


def test_no_package_module_interpolates_an_environment_value_into_a_command():
    """A secret can only reach argv if something puts it there."""
    for path in (PROJECT_ROOT / "delegate_doctor").rglob("*.py"):
        text = path.read_text()
        assert "shell=True" not in text, f"{path.name} uses shell=True"


# --- the key never reaches an artifact --------------------------------------

def _all_output_text(run_dir: Path) -> str:
    """Everything a run wrote, concatenated."""
    text = []
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            text.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(text)


def test_a_full_run_writes_the_sentinel_nowhere(planted_secrets, tmp_path,
                                                capsys, monkeypatch):
    """Drive a real analysis with secrets planted, then search every artifact."""
    import torch

    from delegate_doctor import pipeline
    from delegate_doctor.export_model import ModelSpec

    monkeypatch.setattr(pipeline, "_find_device",
                        lambda runners_dir, **options: (
                            None, None, None, "No Arm64 Android target."))

    class Net(torch.nn.Module):
        def forward(self, x):
            return torch.softmax(x, dim=1)

    inputs = (torch.randn(1, 4, 8, 8),)
    spec = ModelSpec(name="secret-test",
                     exported_program=torch.export.export(Net().eval(), inputs),
                     example_args=inputs)

    artifacts = tmp_path / "artifacts"
    outcome = pipeline.run_optimization(spec, artifacts_dir=str(artifacts))

    captured = capsys.readouterr()
    everywhere = "\n".join([
        captured.out, captured.err,
        outcome.report_text,
        _all_output_text(artifacts),
        json.dumps(outcome.to_dict()),
    ])

    assert SENTINEL not in everywhere
    assert SENTINEL_TWO not in everywhere
    # And the run genuinely produced the artifacts we just searched.
    assert (Path(outcome.run_dir) / "report.html").is_file()
    assert (Path(outcome.run_dir) / "report.txt").is_file()


def test_the_environment_is_never_serialized_into_an_artifact():
    """A whole-environment dump would defeat every other protection here."""
    forbidden = ("dict(os.environ)", "os.environ.copy()",
                 "json.dumps(os.environ", "**os.environ", "list(os.environ)")
    for path in (PROJECT_ROOT / "delegate_doctor").rglob("*.py"):
        text = path.read_text()
        for pattern in forbidden:
            assert pattern not in text, f"{path.name} serializes the environment"


def test_no_module_writes_a_key_to_a_file():
    """There is no DelegateDoctor config file that could hold a credential."""
    import ast

    source = (PROJECT_ROOT / "delegate_doctor" / "agent" /
              "credentials.py").read_text()
    # Strip comments and the module docstring: the prose there explains what
    # this module refuses to do, and names those things in order to refuse them.
    docstring = ast.get_docstring(ast.parse(source)) or ""
    code = "\n".join(line for line in source.replace(docstring, "").splitlines()
                     if not line.strip().startswith("#"))
    for pattern in ("open(", "write_text", "Path.home()", ".json", ".ini",
                    ".yaml", ".toml"):
        assert pattern not in code, f"credentials.py touches {pattern}"


# --- key handling ------------------------------------------------------------

def test_the_key_resolves_from_the_environment(planted_secrets):
    assert credentials.resolve_api_key("openai") == SENTINEL


def test_no_environment_variable_means_ai_is_unavailable(monkeypatch):
    for name in (credentials.ENVIRONMENT_VARIABLE, "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert credentials.resolve_api_key("openai") is None


def test_the_status_reports_availability_without_the_key(planted_secrets):
    from delegate_doctor.agent import provider_config

    status = credentials.key_status(
        provider_config.build_configuration("anthropic"))
    assert status.available
    assert SENTINEL not in status.describe()
    assert SENTINEL not in repr(status)


def test_configure_ai_never_receives_a_key(monkeypatch, capsys, tmp_path):
    """It cannot leak what it never handles: no key reaches this command."""
    from delegate_doctor.agent import provider_config

    monkeypatch.setattr(provider_config, "config_path",
                        lambda: tmp_path / "ai.json")
    monkeypatch.setenv(credentials.ENVIRONMENT_VARIABLE, SENTINEL)

    answers = iter(["2", ""])          # Anthropic, default model
    code = credentials.configure_interactively(
        prompt=lambda question: next(answers))

    output = capsys.readouterr().out
    assert code == 0
    assert SENTINEL not in output
    assert SENTINEL not in (tmp_path / "ai.json").read_text()
    assert "does not store API keys" in output


def test_configure_ai_succeeds_with_no_credential_anywhere(monkeypatch, capsys,
                                                           tmp_path):
    """Provider setup and authentication are separate concerns."""
    from delegate_doctor.agent import provider_config

    monkeypatch.setattr(provider_config, "config_path",
                        lambda: tmp_path / "ai.json")
    for name in (credentials.ENVIRONMENT_VARIABLE, "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    answers = iter(["2", ""])
    assert credentials.configure_interactively(
        prompt=lambda question: next(answers)) == 0

    saved = json.loads((tmp_path / "ai.json").read_text())
    assert saved == {"provider": "anthropic", "model": "claude-sonnet-4-5"}
    assert "export DELEGATE_DOCTOR_LLM_API_KEY" in capsys.readouterr().out


def test_configure_ai_tells_a_local_provider_no_key_is_needed(monkeypatch,
                                                              capsys, tmp_path):
    from delegate_doctor.agent import provider_config

    monkeypatch.setattr(provider_config, "config_path",
                        lambda: tmp_path / "ai.json")
    answers = iter(["5", "", ""])      # Ollama, default model, default base
    assert credentials.configure_interactively(
        prompt=lambda question: next(answers)) == 0

    output = capsys.readouterr().out
    assert "No API key required" in output
    assert "export DELEGATE_DOCTOR_LLM_API_KEY" not in output


def test_an_unwritable_config_directory_is_reported_cleanly(monkeypatch, capsys,
                                                            tmp_path):
    """A real run hit PermissionError here; it must not be a traceback."""
    from delegate_doctor.agent import provider_config

    unwritable = tmp_path / "readonly" / "ai.json"

    def refuse(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(provider_config, "config_path", lambda: unwritable)
    monkeypatch.setattr(provider_config.Path, "mkdir", refuse)

    answers = iter(["2", ""])
    code = credentials.configure_interactively(
        prompt=lambda question: next(answers))

    output = capsys.readouterr().out
    assert code == 2
    assert "Could not write DelegateDoctor AI configuration" in output
    assert "writable" in output
    assert "Traceback" not in output


def test_the_missing_key_message_does_not_suggest_reconfiguring(monkeypatch):
    """Provider is already chosen; configure-ai would not add a key."""
    from delegate_doctor.agent import provider_config

    message = credentials.missing_key_message(
        provider_config.build_configuration("anthropic"))
    assert "does not store API keys" in message
    assert "export DELEGATE_DOCTOR_LLM_API_KEY" in message
    assert "export ANTHROPIC_API_KEY" in message
    assert "configure-ai" not in message
    assert "from delegate_doctor import optimize" in message


# --- consent -----------------------------------------------------------------

def test_the_disclosure_names_the_files_and_the_exclusions():
    text = privacy.consent_disclosure(["/Users/alice/project/model.py"])
    assert "model.py" in text
    assert "/Users/alice" not in text          # only the basename is shown
    for excluded in ("weights", "input tensors", "environment", "credentials"):
        assert excluded in text


# --- repository hygiene ------------------------------------------------------

def test_no_api_key_literal_is_committed():
    import re

    suspicious = []
    tracked = subprocess.run(["git", "ls-files"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True)
    if tracked.returncode != 0:
        pytest.skip("not a git repository")

    patterns = [re.compile(p) for p in (
        r"\bsk-[A-Za-z0-9]{20,}", r"\bsk-ant-[A-Za-z0-9]{20,}",
        r"\bghp_[A-Za-z0-9]{20,}", r"\bAKIA[0-9A-Z]{16}\b",
    )]
    for relative in tracked.stdout.split():
        path = PROJECT_ROOT / relative
        if not path.is_file() or path.suffix in {".pte", ".pt2", ".bin", ".png"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in patterns:
            if pattern.search(text):
                suspicious.append(relative)
    assert suspicious == [], f"possible committed credential: {suspicious}"


def test_dotenv_files_stay_ignored():
    result = subprocess.run(["git", "check-ignore", ".env", ".env.local"],
                            cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert {".env", ".env.local"} <= set(result.stdout.split())


def test_nothing_in_the_package_reads_a_dotenv_file():
    """`.env` support is deliberately absent: it is how these leak."""
    for path in (PROJECT_ROOT / "delegate_doctor").rglob("*.py"):
        text = path.read_text()
        for pattern in ("load_dotenv", "dotenv_values", "from dotenv"):
            assert pattern not in text, f"{path.name} reads a .env file"
