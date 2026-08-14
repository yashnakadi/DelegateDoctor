"""Is this environment actually able to run DelegateDoctor?

The point of this module is to fail in the first two seconds rather than after
a model has been exported, lowered, pushed to a phone and benchmarked. Every
check here is local, fast, and answers a question some real run has stopped on.

Two of them exist because of specific failures, and both are worth naming:

**The Inspector, not just ExecuTorch.** `import executorch` succeeding proves
almost nothing. ETDump analysis goes through
`executorch.devtools.Inspector` -> pandas -> NumPy, and a pandas binary built
against NumPy 1.x on a machine running NumPy 2.x raises

    numpy.core.multiarray failed to import

*at the point the trace is read* - which is after the device work. So the check
imports the real Inspector, which is the only thing that proves the profiling
path will work.

**Where the code actually is.** An editable install pointing at a checkout that
has since moved or been deleted produces confusing failures far from the cause,
so the console script's location and the imported package's location are both
reported and compared.

Nothing here touches the network, a device, or a provider.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Optional

PASS = "PASS"
FAIL = "FAIL"
MISSING = "MISSING"
UNKNOWN = "UNKNOWN"

# The interpreter this project is developed and validated on. Kept in step with
# `requires-python` in pyproject.toml.
SUPPORTED_PYTHON = (3, 12)

# What to tell someone whose environment is not the supported one. One command,
# not a list of repairs.
INSTALL_COMMAND = 'python -m pip install -e ".[ai,examples]"'


@dataclass
class CheckResult:
    """One environment fact, and whether it is good enough to run on."""

    name: str
    status: str
    detail: str = ""
    remedy: str = ""
    optional: bool = False

    @property
    def ok(self) -> bool:
        return self.status == PASS

    @property
    def blocking(self) -> bool:
        """Would this stop an actual optimize run?"""
        return not self.ok and not self.optional

    def format(self) -> str:
        return f"{self.name:<24}{self.status:<9}{self.detail}"

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status,
                "detail": self.detail, "optional": self.optional}


@dataclass
class EnvironmentReport:
    """Everything the preflight learned, in the order it is shown."""

    results: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def add(self, result: CheckResult) -> CheckResult:
        self.results.append(result)
        return result

    def get(self, name: str) -> Optional[CheckResult]:
        return next((item for item in self.results if item.name == name), None)

    @property
    def blocking(self) -> list:
        return [item for item in self.results if item.blocking]

    @property
    def ok(self) -> bool:
        return not self.blocking

    def format(self, verbose: bool = False) -> str:
        lines = ["", "DelegateDoctor environment", ""]
        lines += [item.format() for item in self.results]

        remedies = [item for item in self.blocking if item.remedy]
        if remedies:
            lines.append("")
            for item in remedies:
                lines.append(item.remedy)

        if verbose:
            lines += ["", "Diagnostics", ""]
            for key, value in self.diagnostics.items():
                lines.append(f"{key:<24}{value}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"results": [item.to_dict() for item in self.results],
                "diagnostics": dict(self.diagnostics), "ok": self.ok}


# --- individual checks ----------------------------------------------------------


def _version_of(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return ""
    return str(getattr(module, "__version__", "") or "")


def check_python() -> CheckResult:
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info[:2] == SUPPORTED_PYTHON:
        return CheckResult("Python", PASS, version)
    wanted = ".".join(str(part) for part in SUPPORTED_PYTHON)
    return CheckResult(
        "Python", FAIL, version,
        remedy=(f"DelegateDoctor is validated on Python {wanted}. Create a "
                f"virtual environment on {wanted} and reinstall:\n"
                f"\n    python{wanted} -m venv .venv"
                f"\n    source .venv/bin/activate"
                f"\n    {INSTALL_COMMAND}"))


def check_import(name: str, label: str, remedy: str = "",
                 optional: bool = False) -> CheckResult:
    """One importable dependency, reported with its version."""
    try:
        module = importlib.import_module(name)
    except ImportError as error:
        return CheckResult(label, MISSING, _short(error),
                           remedy=remedy, optional=optional)
    except Exception as error:
        # Not a missing package - an *installed but broken* one, which is the
        # more confusing case and deserves the actual message.
        return CheckResult(label, FAIL, _short(error),
                           remedy=remedy, optional=optional)
    return CheckResult(label, PASS,
                       str(getattr(module, "__version__", "") or ""),
                       optional=optional)


def check_inspector() -> CheckResult:
    """The real ETDump analysis path, not just `import executorch`.

    `optimize` reads traces through this exact import. Checking anything
    shallower would pass on an environment where profiling cannot work.
    """
    try:
        from executorch.devtools import Inspector          # noqa: F401
    except ImportError as error:
        return CheckResult(
            "ETDump Inspector", FAIL, _short(error),
            remedy=_dependency_remedy(error))
    except Exception as error:
        return CheckResult(
            "ETDump Inspector", FAIL, _short(error),
            remedy=_dependency_remedy(error))
    return CheckResult("ETDump Inspector", PASS)


def _dependency_remedy(error: Exception) -> str:
    """Say which known breakage this is, when the message identifies one."""
    text = str(error).lower()
    if "numpy" in text and ("multiarray" in text or "compiled using numpy"
                            in text or "abi" in text):
        return (
            "Installed NumPy and pandas are binary-incompatible: pandas was\n"
            "built against a different NumPy than the one installed.\n"
            "\n"
            "This does not happen in a clean virtual environment. It is\n"
            "usually a conda-installed pandas in an environment where pip\n"
            "later upgraded NumPy.\n"
            "\n"
            "Install DelegateDoctor into its own virtual environment:\n"
            "\n"
            "    python3.12 -m venv .venv\n"
            "    source .venv/bin/activate\n"
            f"    {INSTALL_COMMAND}")
    return (f"The ETDump analysis path could not be imported. Reinstall the\n"
            f"supported stack:\n\n    {INSTALL_COMMAND}")


def _short(error: Exception) -> str:
    text = str(error).strip().splitlines()
    return (text[0] if text else type(error).__name__)[:90]


def describe_path(path) -> str:
    """A path fit to be pasted into an issue: home shortened to `~`.

    Reuses `agent.privacy.redact_home_paths`, which is the existing convention
    for this, so there is one answer to "how do we show a path" rather than
    two that can drift.
    """
    from .agent.privacy import redact_home_paths

    return redact_home_paths(str(path))


def check_adb(path_lookup=shutil.which) -> CheckResult:
    """The adb DelegateDoctor will actually run - resolved, not assumed.

    This deliberately does not consult PATH first. Android Studio installs adb
    at `<sdk>/platform-tools/adb` and does not put it on PATH, so a check that
    asked the shell would report MISSING on a machine that is in fact ready.

    The path is reported through `describe_path`, which shortens the home
    directory to `~`: `check --verbose` output gets pasted into issues, and it
    does not need to carry a username.
    """
    from . import device

    resolved = device.resolve_adb(path_lookup=path_lookup)
    if resolved:
        return CheckResult("ADB", PASS)
    return CheckResult(
        "ADB", MISSING, "",
        remedy=("adb was not found. Install Android Studio, complete its "
                "initial Setup Wizard, then run:\n\n"
                "    delegate-doctor setup-android"))


def check_android_sdk() -> CheckResult:
    from . import android_environment

    root = android_environment.find_sdk_root()
    if root is not None:
        return CheckResult("Android SDK", PASS, str(root))
    return CheckResult(
        "Android SDK", MISSING, "",
        remedy=("No Android SDK was found.\n"
                "\n"
                "Install Android Studio and complete its initial Setup "
                "Wizard.\nThen rerun:\n\n"
                "    delegate-doctor setup-android"))


def check_ai() -> list:
    """AI is optional. A missing provider is never a blocking failure."""
    results = [check_import("litellm", "AI support",
                            remedy=('Install AI support with:\n\n'
                                    '    python -m pip install -e ".[ai]"'),
                            optional=True)]
    try:
        from .agent import provider_config

        configuration = provider_config.load_configuration()
    except Exception:
        configuration = None

    if configuration is None:
        results.append(CheckResult(
            "Provider", MISSING, "not configured",
            remedy="Choose one with:\n\n    delegate-doctor configure-ai",
            optional=True))
    else:
        # Deliberately does not resolve a credential: this is an environment
        # check, and reading a key to report on it is not its business.
        results.append(CheckResult("Provider", PASS, configuration.describe(),
                                   optional=True))
    return results


# --- the whole preflight -----------------------------------------------------------


def run(include_ai: bool = True, include_android: bool = True) -> EnvironmentReport:
    """Every check, in the order the output shows them."""
    report = EnvironmentReport()

    report.add(check_python())
    report.add(check_import("torch", "PyTorch", remedy=_reinstall("PyTorch")))
    report.add(check_import("executorch", "ExecuTorch",
                            remedy=_reinstall("ExecuTorch")))
    report.add(check_import(
        "torchvision", "TorchVision", optional=True,
        remedy=('The bundled examples need TorchVision. Install it with:\n\n'
                '    python -m pip install -e ".[examples]"')))
    report.add(check_import("numpy", "NumPy", remedy=_reinstall("NumPy")))
    report.add(check_import("pandas", "pandas", remedy=_reinstall("pandas")))
    report.add(check_inspector())

    if include_android:
        report.add(check_adb())
        report.add(check_android_sdk())
    if include_ai:
        for result in check_ai():
            report.add(result)

    report.diagnostics = collect_diagnostics()
    return report


def _reinstall(what: str) -> str:
    return f"{what} is missing or broken. Reinstall:\n\n    {INSTALL_COMMAND}"


# The subset an actual optimize run depends on. Checked before export,
# lowering, upload or benchmark, so a dependency problem costs seconds rather
# than the whole device round trip.
DEVICE_CRITICAL = ("Python", "PyTorch", "ExecuTorch", "NumPy", "pandas",
                   "ETDump Inspector")


def preflight(names=DEVICE_CRITICAL) -> EnvironmentReport:
    """The fast subset `optimize` runs before doing anything expensive."""
    report = EnvironmentReport()
    full = run(include_ai=False, include_android=False)
    for result in full.results:
        if result.name in names:
            report.add(result)
    report.diagnostics = full.diagnostics
    return report


def format_preflight_failure(report: EnvironmentReport,
                             verbose: bool = False) -> str:
    """The concise message a blocked run prints. No traceback unless asked."""
    lines = ["", "Environment check          FAILED", ""]
    for result in report.blocking:
        lines.append(f"{result.name:<27}{result.detail or 'unavailable'}")
    for result in report.blocking:
        if result.remedy:
            lines += ["", result.remedy]
    if verbose:
        lines += ["", "Diagnostics", ""]
        for key, value in report.diagnostics.items():
            lines.append(f"{key:<24}{value}")
    return "\n".join(lines)


# --- diagnostics --------------------------------------------------------------------


def collect_diagnostics() -> dict:
    """Paths and versions, for `--verbose`. Makes a stale install obvious."""
    from . import android_environment

    diagnostics = {
        "python executable": sys.executable,
        "python version": sys.version.split()[0],
        "delegate-doctor script": _script_path(),
        "delegate_doctor package": _package_path(),
        "delegate_doctor version": _distribution_version(),
        "torch": _version_of("torch"),
        "torchvision": _version_of("torchvision"),
        "executorch": _version_of("executorch"),
        "numpy": _version_of("numpy"),
        "pandas": _version_of("pandas"),
        "litellm": _version_of("litellm"),
    }

    try:
        environment = android_environment.detect()
        diagnostics["android sdk"] = str(environment.sdk_root or "not found")
        # adb comes from the same resolver the device code uses, so --verbose
        # names the executable that will actually run rather than a second
        # opinion about where adb might be.
        from . import device

        resolved_adb = device.resolve_adb()
        diagnostics["adb"] = (describe_path(resolved_adb) if resolved_adb
                              else "not found")
        for tool in ("sdkmanager",):
            path = environment.tool_path(tool)
            diagnostics[tool] = describe_path(path) if path else "not found"
        diagnostics["ndk"] = (describe_path(environment.ndk)
                              if environment.ndk else "not found")
        diagnostics["cmake"] = (describe_path(environment.cmake)
                                if environment.cmake else "not found")
        diagnostics["git"] = str(environment.git or "not found")
    except Exception:                                    # pragma: no cover
        diagnostics["android sdk"] = "could not be inspected"

    from .pipeline import DEFAULT_RUNNERS_DIR

    diagnostics["runners dir"] = DEFAULT_RUNNERS_DIR
    for name in ("executor_runner_etdump", "executor_runner_bench"):
        path = os.path.join(DEFAULT_RUNNERS_DIR, name)
        diagnostics[name] = path if os.path.isfile(path) else "not built"

    mismatch = describe_install_mismatch()
    if mismatch:
        diagnostics["install warning"] = mismatch
    return diagnostics


def _script_path() -> str:
    path = shutil.which("delegate-doctor")
    return path or "not on PATH"


def _package_path() -> str:
    import delegate_doctor

    return os.path.dirname(os.path.abspath(delegate_doctor.__file__))


def _distribution_version() -> str:
    try:
        from importlib.metadata import version

        return version("delegate-doctor")
    except Exception:
        return "unknown"


def describe_install_mismatch() -> str:
    """Is the console script from a different checkout than the import?

    A stale editable install is the kind of problem that produces symptoms
    nowhere near its cause - code that was definitely edited having no effect -
    so it is worth saying plainly rather than leaving to be deduced.
    """
    script = shutil.which("delegate-doctor")
    if not script:
        return ""

    package = _package_path()
    # The script lives in `<env>/bin`; the package it will import lives under
    # the checkout it was installed from. Comparing prefixes is enough to spot
    # a script from one environment importing a package from another.
    script_env = os.path.dirname(os.path.dirname(os.path.abspath(script)))
    if package.startswith(script_env):
        return ""
    if os.path.commonpath([script_env, package]) == script_env:
        return ""
    return (f"the delegate-doctor script in {script_env} imports "
            f"delegate_doctor from {package}")
