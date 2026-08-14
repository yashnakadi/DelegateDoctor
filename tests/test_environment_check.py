"""Onboarding: does this machine work, and if not, exactly what is wrong?

Two real failures shaped this file.

`import executorch` succeeding proved nothing: ETDump analysis goes through
`executorch.devtools.Inspector` -> pandas -> NumPy, and a pandas built against
NumPy 1.x on NumPy 2.x raised `numpy.core.multiarray failed to import` *while
reading the trace* - after the export, the lowering, the upload and the
benchmark. So the check imports the real Inspector, and `optimize` runs that
subset before it does anything expensive.

And an editable install pointing at a checkout that had moved produced symptoms
nowhere near its cause, so the script's location and the package's location are
compared and reported.

Fully offline: no network, no device, no provider, no browser.
"""

import sys
from pathlib import Path

import pytest

from delegate_doctor import cli, environment_check
# Captured at import: conftest replaces the module attribute so the rest of the
# suite does not depend on this machine's dependency health.
REAL_PREFLIGHT = environment_check.preflight
from delegate_doctor.environment_check import (FAIL, MISSING, PASS,
                                               CheckResult, EnvironmentReport)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --- the report ------------------------------------------------------------------

def test_a_supported_environment_passes(monkeypatch):
    """Case 1: everything importable and the SDK present."""
    report = environment_check.run(include_ai=False, include_android=False)
    blocking = [item.name for item in report.blocking]
    # torch and executorch are installed in the test environment by definition.
    assert "PyTorch" not in blocking
    assert "ExecuTorch" not in blocking


def test_the_python_version_is_checked_against_the_supported_one():
    result = environment_check.check_python()
    assert result.status == (PASS if sys.version_info[:2] ==
                             environment_check.SUPPORTED_PYTHON else FAIL)
    assert result.detail.startswith(str(sys.version_info[0]))


def test_a_missing_package_is_reported_with_a_remedy(monkeypatch):
    result = environment_check.check_import(
        "definitely_not_installed_xyz", "Imaginary",
        remedy="install it")
    assert result.status == MISSING
    assert result.remedy == "install it"


def test_an_installed_but_broken_package_is_not_reported_as_missing(monkeypatch):
    """Case 4: the ABI failure is a *broken* install, not an absent one."""
    def explode(name):
        raise ImportError("numpy.core.multiarray failed to import")

    monkeypatch.setattr(environment_check.importlib, "import_module", explode)
    result = environment_check.check_import("pandas", "pandas")
    assert result.status == MISSING          # ImportError subclass
    assert "multiarray" in result.detail


def test_missing_torchvision_is_not_a_blocking_failure():
    """Case 9: core-only installs do not need the examples extra."""
    result = environment_check.check_import(
        "definitely_not_installed_xyz", "TorchVision", optional=True,
        remedy='python -m pip install -e ".[examples]"')
    assert result.optional
    assert not result.blocking


def test_ai_is_optional_and_never_blocks():
    """Case 9: AI absent is a complete environment, not a broken one."""
    for result in environment_check.check_ai():
        assert result.optional
        assert not result.blocking


def test_the_check_reads_no_credential(monkeypatch):
    """An environment check has no business resolving an API key."""
    from delegate_doctor.agent import credentials

    monkeypatch.setattr(credentials, "resolve_api_key",
                        lambda provider="": pytest.fail("a credential was read"))
    environment_check.check_ai()


# --- the ETDump path, specifically -------------------------------------------------

def test_the_inspector_import_is_what_is_checked():
    """Case 3: not `import executorch` - the path optimize actually uses."""
    import inspect

    source = inspect.getsource(environment_check.check_inspector)
    assert "from executorch.devtools import Inspector" in source


def test_an_abi_failure_gets_the_dependency_remedy():
    """Case 4: the message names the real cause and the real fix."""
    error = ImportError(
        "A module that was compiled using NumPy 1.x cannot be run in "
        "NumPy 2.2.6 ... numpy.core.multiarray failed to import")
    remedy = environment_check._dependency_remedy(error)
    assert "binary-incompatible" in remedy
    assert "virtual environment" in remedy
    assert 'pip install -e ".[ai,examples]"' in remedy
    # And it does not tell the user to hand-repair NumPy.
    assert "pip install numpy<2" not in remedy
    assert "pip install 'numpy<2'" not in remedy


def test_an_unrecognised_import_failure_still_gets_the_install_command():
    remedy = environment_check._dependency_remedy(ImportError("something else"))
    assert environment_check.INSTALL_COMMAND in remedy


def test_a_broken_inspector_blocks(monkeypatch):
    monkeypatch.setattr(
        environment_check, "check_inspector",
        lambda: CheckResult("ETDump Inspector", FAIL,
                            "numpy.core.multiarray failed to import",
                            remedy="reinstall"))
    report = REAL_PREFLIGHT()
    assert not report.ok
    assert any(item.name == "ETDump Inspector" for item in report.blocking)


# --- optimize fails early ------------------------------------------------------------

def test_optimize_runs_the_preflight_before_anything_expensive(monkeypatch,
                                                               capsys):
    """Case 8: no export, no lowering, no device work on a broken environment."""
    broken = EnvironmentReport()
    broken.add(CheckResult("ETDump Inspector", FAIL, "unavailable",
                           remedy="reinstall the supported stack"))
    monkeypatch.setattr(environment_check, "preflight", lambda *a, **k: broken)

    reached = []
    monkeypatch.setattr(cli, "prepare_model_source",
                        lambda *args, **kwargs: reached.append("prepared"))
    monkeypatch.setattr(cli.pipeline, "run_optimization",
                        lambda *args, **kwargs: reached.append("pipeline"))
    monkeypatch.setattr(cli.model_source, "resolve_model_input",
                        lambda target, root=".": _stub_resolved())

    assert cli.run_optimize("model.py", runners_dir=None) == 2
    assert reached == [], "expensive work ran on a broken environment"

    error = capsys.readouterr().err
    assert "Environment check          FAILED" in error
    assert "ETDump Inspector" in error
    assert "reinstall the supported stack" in error


def test_the_preflight_failure_is_concise_without_verbose(monkeypatch):
    broken = EnvironmentReport()
    broken.add(CheckResult("ETDump Inspector", FAIL, "unavailable",
                           remedy="reinstall"))
    broken.diagnostics = {"python executable": "/somewhere/python"}

    terse = environment_check.format_preflight_failure(broken, verbose=False)
    assert "Diagnostics" not in terse
    assert "/somewhere/python" not in terse

    detailed = environment_check.format_preflight_failure(broken, verbose=True)
    assert "Diagnostics" in detailed
    assert "/somewhere/python" in detailed


def test_the_preflight_covers_the_device_critical_subset():
    assert "ETDump Inspector" in environment_check.DEVICE_CRITICAL
    assert "pandas" in environment_check.DEVICE_CRITICAL
    assert "NumPy" in environment_check.DEVICE_CRITICAL
    # Android and AI are not preflight concerns: a target problem is reported
    # by target selection, and AI is optional.
    assert "Android SDK" not in environment_check.DEVICE_CRITICAL
    assert "AI support" not in environment_check.DEVICE_CRITICAL


def _stub_resolved():
    class Resolved:
        path = Path("model.py")
        kind = "python"
        from_workspace = False

    return Resolved()


# --- diagnostics ------------------------------------------------------------------------

def test_verbose_shows_versions_and_paths():
    """Case 6."""
    report = environment_check.run(include_ai=False, include_android=False)
    text = report.format(verbose=True)
    assert "Diagnostics" in text
    for key in ("python executable", "delegate_doctor package", "torch",
                "numpy", "pandas", "runners dir"):
        assert key in text, key


def test_normal_output_hides_the_diagnostics():
    """Case 7."""
    report = environment_check.run(include_ai=False, include_android=False)
    text = report.format(verbose=False)
    assert "Diagnostics" not in text
    assert sys.executable not in text


def test_a_stale_editable_install_is_detected(monkeypatch):
    """Case 5: script from one environment, package from another."""
    monkeypatch.setattr(environment_check.shutil, "which",
                        lambda name: "/opt/other-env/bin/delegate-doctor")
    monkeypatch.setattr(environment_check, "_package_path",
                        lambda: "/somewhere/else/delegate_doctor")

    warning = environment_check.describe_install_mismatch()
    assert warning
    assert "/opt/other-env" in warning
    assert "/somewhere/else" in warning


def test_a_consistent_install_reports_no_mismatch(monkeypatch, tmp_path):
    environment = tmp_path / "venv"
    (environment / "bin").mkdir(parents=True)
    package = environment / "lib" / "delegate_doctor"
    package.mkdir(parents=True)

    monkeypatch.setattr(environment_check.shutil, "which",
                        lambda name: str(environment / "bin" / "delegate-doctor"))
    monkeypatch.setattr(environment_check, "_package_path", lambda: str(package))
    assert environment_check.describe_install_mismatch() == ""


# --- the CLI command ----------------------------------------------------------------------

def test_the_check_command_exists_and_reports(capsys):
    exit_code = cli.main(["check"])
    output = capsys.readouterr().out
    assert "DelegateDoctor environment" in output
    assert exit_code in (0, 2)


def test_the_check_command_takes_verbose(capsys):
    cli.main(["check", "--verbose"])
    assert "Diagnostics" in capsys.readouterr().out


def test_a_failing_check_exits_nonzero(monkeypatch, capsys):
    broken = EnvironmentReport()
    broken.add(CheckResult("pandas", FAIL, "boom", remedy="reinstall"))
    monkeypatch.setattr(environment_check, "run", lambda **kwargs: broken)

    assert cli.main(["check"]) == 2
    assert "Environment ready." not in capsys.readouterr().out


def test_check_is_documented_in_the_help(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "check" in capsys.readouterr().out


# --- packaging -----------------------------------------------------------------------------

def read_pyproject() -> str:
    return (PROJECT_ROOT / "pyproject.toml").read_text()


def test_the_numpy_pandas_floors_are_declared():
    """Case 4/D: packaging makes the broken pair unresolvable."""
    text = read_pyproject()
    assert "numpy>=1.26" in text
    assert "pandas>=2.2.2" in text


def test_the_dependency_groups_are_the_documented_ones():
    """Case 10."""
    text = read_pyproject()
    for group in ("ai = [", "examples = [", "dev = [", "all = ["):
        assert group in text, group
    assert "torchvision==0.28.0" in text
    assert "litellm==1.96.2" in text


def test_the_core_install_requires_neither_ai_nor_examples():
    """Case 9: core stays small."""
    text = read_pyproject()
    core = text.split("[project.optional-dependencies]")[0]
    assert "torchvision" not in core
    assert "litellm" not in core
    assert "segmentation_models_pytorch" not in core


def test_torch_and_executorch_are_pinned_together():
    """Case C: a fresh environment cannot resolve an arbitrary combination."""
    text = read_pyproject()
    assert "executorch==1.4.0" in text
    assert 'requires-python = ">=3.12,<3.13"' in text


# --- the README tells one story ---------------------------------------------------------------

def read_readme() -> str:
    return (PROJECT_ROOT / "README.md").read_text()


def test_the_readme_does_not_ask_users_to_repair_numpy():
    """Case 34."""
    text = read_readme().lower()
    for banned in ("pip install numpy<2", "pip install 'numpy<2'",
                   'pip install "numpy<2"', "downgrade numpy"):
        assert banned not in text, banned


def test_the_readme_does_not_ask_users_to_bootstrap_an_android_sdk():
    """Case 33."""
    text = read_readme().lower()
    for banned in ("commandlinetools", "command-line tools archive",
                   "private android sdk", "sdkmanager --install"):
        assert banned not in text, banned


def test_the_readme_documents_the_canonical_install():
    """The quick start must not require the AI extra.

    AI repair is opt-in behind `--ai-repair` and AI preparation is only needed
    for a model file without the DelegateDoctor interface, so `[ai]` is not
    part of the normal workflow. The full install is still documented, in the
    section that explains what AI is for.
    """
    text = read_readme()
    assert 'pip install -e ".[examples]"' in text
    assert 'pip install -e ".[ai,examples]"' in text
    assert "delegate-doctor check" in text
    assert "delegate-doctor setup-android" in text
    assert "Android Studio" in text
