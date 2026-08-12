"""The graph boundary: pristine baselines, the CLI surface, and what is gone.

Offline. Nothing here reaches a device - the tests stop at the point where the
pipeline would need adb, which is exactly where the input boundary ends.
"""

import copy
import subprocess
from pathlib import Path

import pytest
import torch

from delegate_doctor import cli, export_model, pt2_input
from delegate_doctor.repairs import ALL_RULES, dd001_softmax

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SoftmaxNet(torch.nn.Module):
    """Carries the DD-001 pattern: softmax on a non-last dimension."""

    def forward(self, x):
        return torch.softmax(x, dim=1)


def build_spec(tmp_path):
    inputs = (torch.randn(1, 4, 8, 8),)
    program = torch.export.export(SoftmaxNet().eval(), inputs)
    model_path = str(tmp_path / "model.pt2")
    inputs_path = str(tmp_path / "inputs.pt")
    torch.export.save(program, model_path)
    torch.save(inputs, inputs_path)
    return pt2_input.load_model_spec(model_path, inputs_path)


# --- the pristine baseline --------------------------------------------------

def test_the_baseline_graph_survives_a_repair(tmp_path):
    """The reference for correctness must not be the thing we mutated."""
    spec = build_spec(tmp_path)
    baseline = spec.exported_program

    with torch.no_grad():
        before = baseline.module()(*spec.example_args)

    # What the pipeline does: repair a deep copy, never the spec's program.
    working_copy = copy.deepcopy(baseline)
    detection = dd001_softmax.detect(working_copy)
    assert detection.applies, "the fixture must actually trigger DD-001"
    assert dd001_softmax.apply(working_copy) > 0

    with torch.no_grad():
        after = baseline.module()(*spec.example_args)
    assert torch.equal(before, after), "the baseline graph was mutated"


def test_the_repaired_graph_is_a_different_object_and_did_change(tmp_path):
    spec = build_spec(tmp_path)
    working_copy = copy.deepcopy(spec.exported_program)
    dd001_softmax.apply(working_copy)

    assert working_copy is not spec.exported_program
    baseline_ops = [n.target for n in spec.exported_program.graph.nodes]
    repaired_ops = [n.target for n in working_copy.graph.nodes]
    assert baseline_ops != repaired_ops


def test_the_repair_preserves_semantics_on_the_same_inputs(tmp_path):
    """The existing correctness machinery, exercised on a .pt2-loaded graph."""
    spec = build_spec(tmp_path)
    working_copy = copy.deepcopy(spec.exported_program)
    dd001_softmax.apply(working_copy)

    with torch.no_grad():
        baseline = spec.exported_program.module()(*spec.example_args)
        repaired = working_copy.module()(*spec.example_args)
    assert torch.allclose(baseline, repaired, atol=1e-6)


def test_lowering_does_not_disturb_the_baseline(tmp_path):
    """`to_edge_transform_and_lower` gets its own copy, so this must hold."""
    spec = build_spec(tmp_path)
    with torch.no_grad():
        before = spec.exported_program.module()(*spec.example_args)

    export_model.lower_with_xnnpack(copy.deepcopy(spec.exported_program))

    with torch.no_grad():
        after = spec.exported_program.module()(*spec.example_args)
    assert torch.equal(before, after)


def test_both_repair_rules_still_accept_a_pt2_loaded_program(tmp_path):
    """Rules are unchanged; they just receive a graph that came off disk."""
    spec = build_spec(tmp_path)
    for rule in ALL_RULES:
        result = rule.detect(copy.deepcopy(spec.exported_program))
        assert hasattr(result, "applies")


# --- specs carry a graph, never an nn.Module --------------------------------

def test_a_spec_carries_an_exported_program_not_a_module(tmp_path):
    spec = build_spec(tmp_path)
    assert isinstance(spec.exported_program, torch.export.ExportedProgram)
    assert not hasattr(spec, "model")


# --- CLI surface ------------------------------------------------------------

def test_optimize_requires_an_inputs_file():
    with pytest.raises(SystemExit) as caught:
        cli.main(["optimize", "model.pt2"])
    assert caught.value.code != 0


@pytest.mark.parametrize("target", [
    "https://github.com/owner/repo",
    "http://example.com/model.pt2",
    "https://huggingface.co/owner/model",
])
def test_urls_are_rejected_by_the_cli(target, capsys):
    assert cli.main(["optimize", target, "--inputs", "inputs.pt"]) == 2
    assert "unsupported model input" in capsys.readouterr().err


def test_a_missing_model_file_exits_two(tmp_path, capsys):
    assert cli.main(["optimize", str(tmp_path / "nope.pt2"),
                     "--inputs", str(tmp_path / "nope.pt")]) == 2
    assert "not found" in capsys.readouterr().err


def test_setup_agent_no_longer_exists():
    with pytest.raises(SystemExit):
        cli.main(["setup-agent"])


def test_refresh_flag_no_longer_exists():
    with pytest.raises(SystemExit):
        cli.main(["optimize", "model.pt2", "--inputs", "inputs.pt", "--refresh"])


def test_model_selection_flag_no_longer_exists():
    with pytest.raises(SystemExit):
        cli.main(["optimize", "model.pt2", "--inputs", "inputs.pt",
                  "--model", "resnet20"])


def test_the_remaining_subcommands_are_the_intended_two(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    text = capsys.readouterr().out
    assert "optimize" in text and "setup-android" in text
    assert "setup-agent" not in text
    assert "doctor MODEL" not in text


def test_help_advertises_the_pt2_workflow(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    text = capsys.readouterr().out
    assert "model.pt2" in text and "--inputs" in text
    for gone in ("github", "GitHub", "Ollama", "repository URL"):
        assert gone not in text


# --- nothing of the removed subsystems remains ------------------------------

REMOVED_TOKENS = (
    "ollama", "Ollama", "qwen", "setup-agent", "MAX_INGESTION_ATTEMPTS",
    "need_user_input", "adapter_code", "127.0.0.1:11434", "litellm", "LiteLLM",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "hubconf", "commit_sha",
    "OWNER__REPO", "github_source", "ingest_github", "create_model()",
)


def test_no_ai_or_repository_ingestion_remains_in_the_package():
    for path in (PROJECT_ROOT / "delegate_doctor").rglob("*.py"):
        text = path.read_text()
        for token in REMOVED_TOKENS:
            assert token not in text, f"{path.name} still references {token}"


def test_the_removed_modules_are_gone():
    for relative in ("delegate_doctor/ingestion", "delegate_doctor/custom_model.py",
                     "delegate_doctor/github_source.py", "delegate_doctor/redaction.py",
                     "examples/custom_model.py"):
        assert not (PROJECT_ROOT / relative).exists(), f"{relative} still exists"


def test_no_module_imports_the_removed_subsystems():
    for path in (PROJECT_ROOT / "delegate_doctor").rglob("*.py"):
        text = path.read_text()
        for token in ("import ollama", "from .ingestion", "import custom_model",
                      "from .custom_model", "import redaction", "urllib.request"):
            assert token not in text, f"{path.name} imports {token}"


def test_the_readme_documents_the_pt2_workflow():
    text = (PROJECT_ROOT / "README.md").read_text()
    assert "torch.export.save" in text
    assert "optimize model.pt2 --inputs inputs.pt" in text
    for gone in ("Ollama", "qwen", "setup-agent", "hubconf", "--refresh"):
        assert gone not in text, f"README still mentions {gone}"


# --- repository hygiene (kept from the previous suite) ----------------------

def test_repository_has_no_committed_credentials():
    """Tracked text files must not contain anything resembling a live key."""
    import re

    patterns = (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
        re.compile(r"(?i)\b(api[_-]?key|secret|password|token)\s*[=:]\s*"
                   r"['\"][A-Za-z0-9._\-]{16,}['\"]"),
    )
    tracked = subprocess.run(["git", "ls-files"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True)
    if tracked.returncode != 0:
        pytest.skip("not a git repository")

    suspicious = []
    for relative in tracked.stdout.split():
        path = PROJECT_ROOT / relative
        if not path.is_file() or path.suffix in {".pte", ".pt2", ".bin", ".png"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in patterns:
            if pattern.search(text):
                suspicious.append(relative)
    assert suspicious == [], f"possible committed credential material: {suspicious}"


def test_gitignore_still_blocks_secret_and_artifact_files():
    result = subprocess.run(
        ["git", "check-ignore", ".env", "private.pem", "server.key",
         "artifacts/run_001", "runners/executor_runner_bench"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    ignored = set(result.stdout.split())
    assert {".env", "private.pem", "server.key"} <= ignored
