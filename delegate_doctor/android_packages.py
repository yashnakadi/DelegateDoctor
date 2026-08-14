"""Inspect and install Android SDK packages, and handle SDK licences.

DelegateDoctor needs two things from the Android SDK: `platform-tools` (adb, to
talk to the phone) and the pinned NDK (to cross-compile the two ExecuTorch
runners). This module is the one place that asks sdkmanager what is installed
and installs what is missing.

Every external command runs through `run_tool()` with an argument list - never a
shell string - so nothing here can be influenced by a path containing a space,
a quote, or anything worse.

Licences are never answered on the user's behalf. `accept_licenses_interactively`
hands the terminal to `sdkmanager --licenses` and gets out of the way: the
agreement is between the user and Google.

This module was once `emulator.py`, which also created and booted a managed AVD.
DelegateDoctor is physical-phone-only now, so that half is gone; what remained
was never emulator-specific.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


from .android_environment import AndroidEnvironment

# The NDK the runner build is validated against. Pinned for the same reason the
# ExecuTorch commit is: a different toolchain produces a different binary, and
# the runners must match the ExecuTorch Python package they were validated with.
NDK_VERSION = "27.2.12479018"
NDK_PACKAGE = f"ndk;{NDK_VERSION}"

# Everything DelegateDoctor needs from the SDK. Deliberately two entries: adb
# to reach the phone, and the pinned NDK to cross-compile the runners. There is
# no emulator, no system image and no SDK platform, because a physical phone
# needs none of them.
REQUIRED_PACKAGES = (
    "platform-tools",
    NDK_PACKAGE,
)

# One cap for every sdkmanager invocation.
COMMAND_TIMEOUT_SECONDS = 120


class AndroidPackageError(RuntimeError):
    """An expected SDK problem, reported as a message rather than a traceback."""


@dataclass
class ToolResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_tool(executable: Path, arguments: list, timeout: int = COMMAND_TIMEOUT_SECONDS,
             input_text: str = None) -> ToolResult:
    """Run one Android tool. Argument list only; never a shell.

    The single choke point for external Android commands, which is what makes
    the rest of this module testable without an SDK.
    """
    command = [str(executable)] + [str(argument) for argument in arguments]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            input=input_text,
        )
    except FileNotFoundError:
        raise AndroidPackageError(f"{Path(executable).name} was not found at {executable}")
    except subprocess.TimeoutExpired:
        raise AndroidPackageError(
            f"{Path(executable).name} did not finish within {timeout}s."
        )
    return ToolResult(completed.returncode, completed.stdout or "",
                      completed.stderr or "")


# --- SDK packages -----------------------------------------------------------


def installed_packages(environment: AndroidEnvironment) -> set:
    """Package IDs sdkmanager reports as installed."""
    sdkmanager = environment.tool_path("sdkmanager")
    if sdkmanager is None:
        raise AndroidPackageError("sdkmanager is not available.")

    result = run_tool(sdkmanager, ["--list_installed"])
    if not result.ok:
        # Older sdkmanager builds spell it differently; try the general form.
        result = run_tool(sdkmanager, ["--list"])
        if not result.ok:
            raise AndroidPackageError(
                "sdkmanager could not list installed packages.\n"
                f"{result.stderr.strip()[:400]}"
            )
    return parse_installed_packages(result.stdout)


def parse_installed_packages(text: str) -> set:
    """Read package IDs out of sdkmanager's table.

    Its output is a bordered table with a header, and a `--list` run continues
    into an "Available Packages" section that must not be counted as installed.
    """
    packages = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("-", "=")):
            continue
        lowered = stripped.lower()
        if lowered.startswith("available packages") or lowered.startswith(
                "available updates"):
            break
        if lowered.startswith(("installed packages", "path ", "path|")):
            continue
        # "  emulator | 35.1.4 | Android Emulator | emulator"
        first = stripped.split("|")[0].strip()
        if first and " " not in first:
            packages.add(first)
    return packages


def missing_packages(environment: AndroidEnvironment,
                     required=REQUIRED_PACKAGES) -> list:
    """Which required packages are not installed, in the declared order."""
    present = installed_packages(environment)
    return [package for package in required if package not in present]


def install_packages(environment: AndroidEnvironment, packages: list,
                     announce=print) -> None:
    """Install exactly the packages named. Nothing else, ever."""
    if not packages:
        return
    sdkmanager = environment.tool_path("sdkmanager")
    if sdkmanager is None:
        raise AndroidPackageError("sdkmanager is not available.")

    for package in packages:
        announce(f"  installing {package}")
        result = run_tool(sdkmanager, ["--install", package],
                          timeout=60 * 60)
        if not result.ok:
            raise AndroidPackageError(
                f"Could not install {package}.\n"
                f"\n"
                f"{(result.stderr or result.stdout).strip()[:600]}\n"
                f"\n"
                f"If this is a licence problem, run:\n"
                f"\n"
                f"    {sdkmanager} --licenses"
            )


def licenses_accepted(environment: AndroidEnvironment) -> bool:
    """Have the SDK licences been accepted?

    Checked rather than assumed: DelegateDoctor will not pipe agreement into a
    legal prompt on the user's behalf.
    """
    sdk_root = environment.sdk_root
    if sdk_root is None:
        return False
    licenses_dir = sdk_root / "licenses"
    if not licenses_dir.is_dir():
        return False
    try:
        return any(child.is_file() for child in licenses_dir.iterdir())
    except OSError:
        return False


LICENSES_MESSAGE = (
    "Android SDK licences have not been accepted.\n"
    "\n"
    "Google requires you - not DelegateDoctor - to read and accept them:\n"
    "\n"
    "    sdkmanager --licenses\n"
    "\n"
    "then run setup again."
)

NONINTERACTIVE_LICENSES_MESSAGE = (
    "Android SDK licences have not been accepted, and this run cannot ask.\n"
    "\n"
    "DelegateDoctor will not answer a legal prompt on your behalf, and there\n"
    "is no flag that makes it do so.\n"
    "\n"
    "Run this once, interactively, and read what it shows you:\n"
    "\n"
    "    delegate-doctor setup-android\n"
    "\n"
    "or accept them directly:\n"
    "\n"
    "    sdkmanager --licenses"
)


def accept_licenses_interactively(environment: AndroidEnvironment,
                                  announce=print, prompt=input,
                                  runner=None) -> bool:
    """Hand the user's terminal to `sdkmanager --licenses`.

    DelegateDoctor starts the process and gets out of the way. It does not
    capture the output, does not answer the prompts, and does not pipe "y" in:
    the licences are a legal agreement between the user and Google, and a tool
    that clicks through them on your behalf has agreed to nothing on your
    behalf that means anything.
    """
    sdkmanager = environment.tool_path("sdkmanager")
    if sdkmanager is None:
        raise AndroidPackageError("sdkmanager is not available.")

    announce("\nAndroid SDK licences must be accepted.\n")
    try:
        prompt("Press Enter to review licences, or Ctrl-C to stop: ")
    except (EOFError, KeyboardInterrupt):
        return False

    # Deliberately *not* run_tool(): that captures stdout, which would leave
    # the user answering prompts they cannot see.
    run = runner or subprocess.call
    try:
        run([str(sdkmanager), "--licenses"])
    except (OSError, subprocess.SubprocessError) as error:
        raise AndroidPackageError(f"Could not run sdkmanager --licenses: {error}")

    return licenses_accepted(environment)
