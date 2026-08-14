"""Path resolution: exactly two places are ever looked at, in a fixed order.

The risk here is a convenience that becomes unpredictable. These tests pin the
rule down: an explicit path is used exactly as typed, a bare filename gets one
documented fallback to `models/`, and nothing else on the filesystem is ever
consulted.
"""

import subprocess
from pathlib import Path

import pytest

from delegate_doctor import model_source
from delegate_doctor.model_source import ModelSourceError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A project root with a models/ directory, and cwd pointed at it."""
    (tmp_path / "models").mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# --- the bare-filename fallback ---------------------------------------------

def test_a_bare_name_resolves_in_the_workspace(workspace):
    write(workspace / "models" / "model.py")
    resolved = model_source.resolve_model_input("model.py", workspace)
    assert resolved.path == (workspace / "models" / "model.py").resolve()
    assert resolved.from_workspace
    assert resolved.is_python


def test_the_current_directory_wins_over_the_workspace(workspace):
    """Both exist: the one the user is standing in is the one they meant."""
    write(workspace / "model.py", "current")
    write(workspace / "models" / "model.py", "workspace")
    resolved = model_source.resolve_model_input("model.py", workspace)
    assert resolved.path.read_text() == "current"
    assert not resolved.from_workspace


def test_a_serialized_program_is_refused_and_names_the_python_api(workspace):
    """.pt2 was a public entry point once. It must not fail as "unsupported"."""
    write(workspace / "models" / "model.pt2")
    with pytest.raises(ModelSourceError) as caught:
        model_source.resolve_model_input("model.pt2", workspace)
    message = str(caught.value)
    assert "serialized artifact" in message
    assert "from delegate_doctor import optimize" in message


def test_a_serialized_input_tuple_is_refused_the_same_way(workspace):
    write(workspace / "models" / "inputs.pt")
    with pytest.raises(ModelSourceError) as caught:
        model_source.resolve_model_input("inputs.pt", workspace)
    assert "from delegate_doctor import optimize" in str(caught.value)


def test_serialized_artifacts_are_refused_before_the_file_is_looked_for():
    """The refusal is about the *kind* of file, so it needs no filesystem."""
    with pytest.raises(ModelSourceError) as caught:
        model_source.resolve_model_input("/nowhere/at/all/model.pt2")
    assert "serialized artifact" in str(caught.value)


# --- explicit paths always win ----------------------------------------------

def test_an_explicit_relative_path_is_used_exactly(workspace):
    write(workspace / "projects" / "foo" / "model.py", "explicit")
    write(workspace / "models" / "model.py", "workspace")
    resolved = model_source.resolve_model_input("projects/foo/model.py", workspace)
    assert resolved.path.read_text() == "explicit"
    assert not resolved.from_workspace


def test_an_explicit_path_is_never_redirected_to_the_workspace(workspace):
    """A typo in an explicit path must not silently find a different file."""
    write(workspace / "models" / "model.py", "workspace")
    with pytest.raises(ModelSourceError) as caught:
        model_source.resolve_model_input("projects/foo/model.py", workspace)
    message = str(caught.value)
    assert "not found" in message
    assert "models" not in message.replace("models/model.py", "")


def test_an_absolute_path_is_used_exactly(workspace, tmp_path):
    elsewhere = write(tmp_path / "elsewhere" / "model.py", "absolute")
    resolved = model_source.resolve_model_input(str(elsewhere), workspace)
    assert resolved.path.read_text() == "absolute"


def test_a_dot_slash_path_is_explicit(workspace):
    write(workspace / "models" / "model.py", "workspace")
    with pytest.raises(ModelSourceError):
        model_source.resolve_model_input("./model.py", workspace)


@pytest.mark.parametrize("target, explicit", [
    ("model.py", False),
    ("models/model.py", True),
    ("./model.py", True),
    ("../model.py", True),
    ("a/b/model.py", True),
    ("/abs/model.py", True),
    ("~/model.py", True),
    ("dir\\model.py", True),
])
def test_explicitness_is_decided_by_the_presence_of_a_directory(target, explicit):
    assert model_source.has_directory_component(target) is explicit


# --- exactly two candidates, never a search ---------------------------------

def test_a_bare_name_has_exactly_two_candidates(workspace):
    candidates = model_source.candidate_paths("model.py", workspace)
    assert len(candidates) == 2
    assert candidates[0] == Path("model.py")
    assert candidates[1] == workspace / "models" / "model.py"


def test_an_explicit_path_has_exactly_one_candidate(workspace):
    assert len(model_source.candidate_paths("a/b/model.py", workspace)) == 1


def test_nested_workspace_files_are_not_discovered(workspace):
    """No recursive search: models/sub/model.py is not found by `model.py`."""
    write(workspace / "models" / "sub" / "model.py")
    with pytest.raises(ModelSourceError):
        model_source.resolve_model_input("model.py", workspace)


def test_a_similarly_named_file_is_not_substituted(workspace):
    write(workspace / "models" / "my_model.py")
    with pytest.raises(ModelSourceError):
        model_source.resolve_model_input("model.py", workspace)


def test_the_failure_message_lists_everywhere_it_looked(workspace):
    with pytest.raises(ModelSourceError) as caught:
        model_source.resolve_model_input("model.py", workspace)
    message = str(caught.value)
    assert "Looked in:" in message
    assert "model.py" in message
    assert str(workspace / "models") in message


# --- rejected inputs ---------------------------------------------------------

@pytest.mark.parametrize("target", [
    "https://github.com/owner/repo",
    "http://example.com/model.py",
    "github.com/owner/repo",
    "git@github.com:owner/repo.git",
])
def test_urls_are_rejected(target):
    with pytest.raises(ModelSourceError) as caught:
        model_source.resolve_model_input(target)
    assert "unsupported model input" in str(caught.value)


def test_a_pte_is_refused_with_the_distinction_spelled_out(workspace):
    with pytest.raises(ModelSourceError) as caught:
        model_source.resolve_model_input("model.pte", workspace)
    message = str(caught.value)
    assert "DelegateDoctor's *output*" in message
    assert ".py" in message


def test_an_unsupported_suffix_is_refused(workspace):
    with pytest.raises(ModelSourceError) as caught:
        model_source.resolve_model_input("model.onnx", workspace)
    assert "Unsupported model file" in str(caught.value)


def test_a_directory_is_refused(workspace):
    (workspace / "models" / "model.py").mkdir(parents=True)
    with pytest.raises(ModelSourceError) as caught:
        model_source.resolve_model_input("model.py", workspace)
    assert "is a directory" in str(caught.value)


def test_an_empty_target_is_refused():
    with pytest.raises(ModelSourceError):
        model_source.resolve_model_input("")


def test_special_files_are_refused(workspace):
    import os

    fifo = workspace / "models" / "model.py"
    try:
        os.mkfifo(fifo)
    except (AttributeError, OSError):
        pytest.skip("this platform cannot create a FIFO")
    with pytest.raises(ModelSourceError) as caught:
        model_source.resolve_model_input("model.py", workspace)
    assert "not a regular file" in str(caught.value)


def test_an_unknown_suffix_names_the_one_supported_input(workspace):
    write(workspace / "models" / "model.bin")
    with pytest.raises(ModelSourceError) as caught:
        model_source.resolve_model_input("model.bin", workspace)
    message = str(caught.value)
    assert "Expected a .py model source" in message
    assert "--inputs" not in message


# --- the CLI honours all of this --------------------------------------------

def test_the_cli_refuses_a_pt2_and_points_at_the_python_api(workspace, capsys):
    """The removed entry point fails with guidance, not a stack trace."""
    from delegate_doctor import cli

    write(workspace / "models" / "model.pt2")
    assert cli.main(["optimize", "model.pt2"]) == 2
    error = capsys.readouterr().err
    assert "serialized artifact" in error
    assert "from delegate_doctor import optimize" in error


def test_the_cli_no_longer_accepts_an_inputs_flag(workspace, capsys):
    """--inputs existed only for the artifact path, so argparse must reject it."""
    from delegate_doctor import cli

    write(workspace / "models" / "model.py", "import torch\n")
    with pytest.raises(SystemExit) as caught:
        cli.main(["optimize", "model.py", "--inputs", "inputs.pt"])
    assert caught.value.code == 2
    assert "--inputs" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--no-ai", "--ai-repairs", "--allow-ai-repair"])
def test_the_old_ai_mode_flags_are_gone(workspace, flag):
    """They existed only to maintain the AI/no-AI split, which is gone."""
    from delegate_doctor import cli

    write(workspace / "models" / "model.py", "import torch\n")
    with pytest.raises(SystemExit):
        cli.main(["optimize", "model.py", flag])


def test_source_without_the_interface_names_both_ways_forward(workspace, capsys):
    """No interface and no provider: say so, and say what to write."""
    from delegate_doctor import cli

    write(workspace / "models" / "model.py", "import torch\n")
    assert cli.main(["optimize", "model.py"]) == 2
    error = capsys.readouterr().err
    assert "model interface not found" in error
    assert "delegate_doctor_model" in error
    assert "delegate_doctor_inputs" in error
    assert "Traceback" not in error


def test_python_source_without_a_provider_says_so_cleanly(workspace, capsys,
                                                          monkeypatch):
    """No key configured: an actionable message, never a raw SDK exception."""
    from delegate_doctor import cli
    from delegate_doctor.agent import credentials

    monkeypatch.delenv(credentials.ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    write(workspace / "models" / "model.py", "import torch\n")
    assert cli.main(["optimize", "model.py"]) == 2
    error = capsys.readouterr().err
    assert "AI preparation is unavailable" in error
    assert "Traceback" not in error


def test_the_cli_reports_a_missing_model_without_a_traceback(workspace, capsys):
    from delegate_doctor import cli

    assert cli.main(["optimize", "model.py"]) == 2
    assert "not found" in capsys.readouterr().err


# --- the workspace is private ------------------------------------------------

def test_the_workspace_exists_with_a_readme():
    assert (PROJECT_ROOT / "models").is_dir()
    assert (PROJECT_ROOT / "models" / "README.md").is_file()


def test_workspace_contents_are_ignored_by_git():
    """A user's model source and weights must never be committed."""
    candidates = ["models/model.py", "models/model.pt2", "models/inputs.pt",
                  "models/weights.pth", "models/private_net.py"]
    result = subprocess.run(["git", "check-ignore", *candidates],
                            cwd=PROJECT_ROOT, capture_output=True, text=True)
    ignored = set(result.stdout.split())
    assert set(candidates) <= ignored, f"not ignored: {set(candidates) - ignored}"


def test_the_workspace_readme_is_tracked():
    result = subprocess.run(["git", "check-ignore", "models/README.md"],
                            cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode != 0, "models/README.md should be committed"


def test_examples_is_not_the_user_workspace():
    """examples/ stays the checked-in demonstration suite."""
    result = subprocess.run(
        ["git", "check-ignore", "examples/dd001_softmax/unet.py"],
        cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode != 0, "examples/ must remain tracked"
