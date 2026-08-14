"""Provision and start the one Arm64 emulator DelegateDoctor manages.

Scope is deliberately narrow. DelegateDoctor owns exactly one AVD, named by the
constant below, and touches nothing else: it never lists-and-deletes, never
wipes, never renames, and never issues a command that affects every emulator on
the machine. A user's own AVDs are none of its business.

    inspect packages -> install missing ones (with consent)
    -> create the AVD if absent -> start it -> wait for boot -> verify the ABI

The last step is not a formality. An emulator that boots is not necessarily an
*Arm* emulator, and a benchmark from an x86_64 image would be a measurement of
the host, so the ABI is confirmed before the target is ever handed back.

Every external command runs through `run_tool()` with an argument list - never a
shell string - so nothing here can be influenced by a path containing a space,
a quote, or anything worse.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


from .android_environment import AndroidEnvironment

# --- the pinned configuration ----------------------------------------------
#
# API 35 and google_apis were chosen because that is the image the project's
# emulator evidence was recorded on. Drifting either would silently change what
# the numbers mean, so both are constants rather than "whatever is newest".

ANDROID_API_LEVEL = 35
SYSTEM_IMAGE_TAG = "google_apis"
SYSTEM_IMAGE_ABI = "arm64-v8a"

AVD_NAME = "DelegateDoctor_ARM64"

SYSTEM_IMAGE_PACKAGE = (
    f"system-images;android-{ANDROID_API_LEVEL};{SYSTEM_IMAGE_TAG};{SYSTEM_IMAGE_ABI}"
)

# The NDK the runner build is validated against. Pinned for the same reason the
# ExecuTorch commit is: a different toolchain is a different binary.
NDK_VERSION = "27.2.12479018"
NDK_PACKAGE = f"ndk;{NDK_VERSION}"

# Everything needed to create and run the managed AVD.
EMULATOR_PACKAGES = (
    "platform-tools",
    "emulator",
    f"platforms;android-{ANDROID_API_LEVEL}",
    SYSTEM_IMAGE_PACKAGE,
)

# Launch flags, kept together so their effect on measurement is reviewable.
#   -no-window / -no-audio  headless; neither touches CPU scheduling
#   -no-snapshot            boot clean rather than restoring a saved state,
#                           whose warmed caches would flatter the first run
# Deliberately absent: -wipe-data (destroys state every run) and any of the
# CPU-throttling or -netdelay options, which would make latency meaningless.
EMULATOR_LAUNCH_FLAGS = ("-no-window", "-no-audio", "-no-snapshot")

BOOT_TIMEOUT_SECONDS = 300
BOOT_POLL_INTERVAL_SECONDS = 2
ADB_APPEAR_TIMEOUT_SECONDS = 120
COMMAND_TIMEOUT_SECONDS = 120


class EmulatorError(RuntimeError):
    """An expected emulator problem, reported as a message rather than a traceback."""


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
        raise EmulatorError(f"{Path(executable).name} was not found at {executable}")
    except subprocess.TimeoutExpired:
        raise EmulatorError(
            f"{Path(executable).name} did not finish within {timeout}s."
        )
    return ToolResult(completed.returncode, completed.stdout or "",
                      completed.stderr or "")


# --- SDK packages -----------------------------------------------------------


def installed_packages(environment: AndroidEnvironment) -> set:
    """Package IDs sdkmanager reports as installed."""
    sdkmanager = environment.tool_path("sdkmanager")
    if sdkmanager is None:
        raise EmulatorError("sdkmanager is not available.")

    result = run_tool(sdkmanager, ["--list_installed"])
    if not result.ok:
        # Older sdkmanager builds spell it differently; try the general form.
        result = run_tool(sdkmanager, ["--list"])
        if not result.ok:
            raise EmulatorError(
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
                     required=EMULATOR_PACKAGES) -> list:
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
        raise EmulatorError("sdkmanager is not available.")

    for package in packages:
        announce(f"  installing {package}")
        result = run_tool(sdkmanager, ["--install", package],
                          timeout=60 * 60)
        if not result.ok:
            raise EmulatorError(
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
        raise EmulatorError("sdkmanager is not available.")

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
        raise EmulatorError(f"Could not run sdkmanager --licenses: {error}")

    return licenses_accepted(environment)


# --- the managed AVD --------------------------------------------------------


def list_avds(environment: AndroidEnvironment) -> list:
    """Existing AVD names, via the emulator's own listing."""
    emulator_path = environment.tool_path("emulator")
    if emulator_path is None:
        return []
    try:
        result = run_tool(emulator_path, ["-list-avds"], timeout=60)
    except EmulatorError:
        # Listing AVDs is a status question. A missing or unusable emulator
        # binary is worth reporting as "no AVDs", not worth crashing a report.
        return []
    if not result.ok:
        return []
    return [line.strip() for line in result.stdout.splitlines()
            if line.strip() and " " not in line.strip()]


def avd_exists(environment: AndroidEnvironment, name: str = AVD_NAME) -> bool:
    return name in list_avds(environment)


def avd_config_path(environment: AndroidEnvironment, name: str = AVD_NAME,
                    home: Path = None) -> Path:
    """Where the AVD's own config lives. Read to check it, never to edit it."""
    base = Path(home) if home is not None else Path.home()
    return base / ".android" / "avd" / f"{name}.avd" / "config.ini"


def avd_is_compatible(environment: AndroidEnvironment, name: str = AVD_NAME,
                      home: Path = None) -> tuple:
    """(compatible, reason) for DelegateDoctor's own AVD.

    An AVD carrying the right name but an x86_64 image would produce numbers
    that are not Arm numbers, which is the one failure this whole project
    exists to prevent. Only the config file is read; nothing is modified, and
    no other AVD is looked at.
    """
    config = avd_config_path(environment, name, home)
    if not config.is_file():
        # No config to disagree with. The AVD listing is the authority on
        # existence, and a missing file here is not evidence of a mismatch.
        return True, ""
    try:
        text = config.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return True, ""

    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key.strip() in ("abi.type", "hw.cpu.arch", "image.sysdir.1"):
            value = value.strip().lower()
            if not value:
                continue
            if SYSTEM_IMAGE_ABI in value or "arm64" in value or value == "arm64":
                return True, ""
            if "x86" in value or "arm" not in value:
                return False, (
                    f"{name} is configured for {value!r}, not "
                    f"{SYSTEM_IMAGE_ABI}. DelegateDoctor measures Arm64, and "
                    f"an x86 emulator result is not Arm evidence."
                )
    return True, ""


def recreate_avd(environment: AndroidEnvironment, name: str = AVD_NAME,
                 announce=print) -> None:
    """Delete and rebuild DelegateDoctor's own AVD. Never any other.

    Guarded by the name: the one AVD this project created is the only one it
    may destroy, and a caller passing something else is a bug worth raising on
    rather than a request worth honouring.
    """
    if name != AVD_NAME:
        raise EmulatorError(
            f"DelegateDoctor only manages {AVD_NAME}. It will not delete "
            f"{name!r}, which it did not create."
        )
    avdmanager = environment.tool_path("avdmanager")
    if avdmanager is None:
        raise EmulatorError("avdmanager is not available.")

    announce(f"  recreating AVD {name}")
    run_tool(avdmanager, ["delete", "avd", "--name", name], timeout=5 * 60)
    create_avd(environment, name, announce=announce)


def create_avd(environment: AndroidEnvironment, name: str = AVD_NAME,
               announce=print) -> None:
    """Create DelegateDoctor's AVD. Never overwrites an existing one.

    `--force` is deliberately not passed: if something already owns this name,
    the right response is to stop and say so, not to overwrite it.
    """
    if avd_exists(environment, name):
        return

    avdmanager = environment.tool_path("avdmanager")
    if avdmanager is None:
        raise EmulatorError("avdmanager is not available.")

    announce(f"  creating AVD {name}")
    result = run_tool(
        avdmanager,
        ["create", "avd",
         "--name", name,
         "--package", SYSTEM_IMAGE_PACKAGE,
         "--abi", SYSTEM_IMAGE_ABI,
         "--device", "pixel_6",
         # Answers avdmanager's "custom hardware profile?" prompt with the
         # default, so the command never blocks waiting on stdin.
         ],
        input_text="no\n",
        timeout=10 * 60,
    )
    if not result.ok:
        raise EmulatorError(
            f"Could not create the AVD {name}.\n"
            f"\n"
            f"{(result.stderr or result.stdout).strip()[:600]}\n"
            f"\n"
            f"Check that {SYSTEM_IMAGE_PACKAGE} is installed."
        )


# --- starting it ------------------------------------------------------------


def _adb_serials(environment: AndroidEnvironment) -> set:
    """Serials adb currently reports, in any state."""
    adb = environment.tool_path("adb")
    if adb is None:
        return set()
    result = run_tool(adb, ["devices"], timeout=30)
    serials = set()
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            serials.add(parts[0])
    return serials


def _adb_property(environment: AndroidEnvironment, serial: str, name: str) -> str:
    adb = environment.tool_path("adb")
    if adb is None:
        return ""
    result = run_tool(adb, ["-s", serial, "shell", "getprop", name], timeout=30)
    return result.stdout.strip()


def launch_emulator_process(environment: AndroidEnvironment,
                            name: str = AVD_NAME) -> subprocess.Popen:
    """Start the emulator without blocking DelegateDoctor.

    The emulator runs for the whole session, so it is never waited on. Its
    output is discarded rather than inherited, so a boot log cannot interleave
    with the report.
    """
    emulator_path = environment.tool_path("emulator")
    if emulator_path is None:
        raise EmulatorError("The Android emulator executable was not found.")

    command = [str(emulator_path), "-avd", name, *EMULATOR_LAUNCH_FLAGS]
    return subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )


def wait_for_new_serial(environment: AndroidEnvironment, before: set,
                        timeout: int = ADB_APPEAR_TIMEOUT_SECONDS,
                        sleep=time.sleep, now=time.monotonic) -> str:
    """The serial that appeared after launching. Never assumes emulator-5554.

    Comparing against the set of serials seen *before* the launch is what stops
    DelegateDoctor from adopting somebody else's already-running emulator.
    """
    deadline = now() + timeout
    while now() < deadline:
        appeared = _adb_serials(environment) - before
        if appeared:
            return sorted(appeared)[0]
        sleep(BOOT_POLL_INTERVAL_SECONDS)
    raise EmulatorError(
        f"The emulator did not appear in `adb devices` within {timeout}s."
    )


def wait_for_boot(environment: AndroidEnvironment, serial: str,
                  timeout: int = BOOT_TIMEOUT_SECONDS,
                  sleep=time.sleep, now=time.monotonic) -> None:
    """Wait until Android has actually finished booting.

    `adb wait-for-device` returns as soon as the daemon answers, long before
    the system is usable, so `sys.boot_completed` is polled instead - with a
    deadline, because an emulator that never boots must fail rather than hang.
    """
    deadline = now() + timeout
    while now() < deadline:
        if _adb_property(environment, serial, "sys.boot_completed") == "1":
            return
        sleep(BOOT_POLL_INTERVAL_SECONDS)
    raise EmulatorError(
        f"The emulator started but did not finish booting within {timeout}s.\n"
        f"\n"
        f"Serial: {serial}\n"
        f"\n"
        f"Try starting it by hand to see what it reports."
    )


def verify_abi(environment: AndroidEnvironment, serial: str) -> str:
    """Confirm the booted emulator really is arm64-v8a."""
    abi = _adb_property(environment, serial, "ro.product.cpu.abi")
    if abi != SYSTEM_IMAGE_ABI:
        raise EmulatorError(
            f"ARM64 EMULATOR UNAVAILABLE\n"
            f"\n"
            f"The emulator booted as {abi or 'an unknown ABI'}, not "
            f"{SYSTEM_IMAGE_ABI}.\n"
            f"\n"
            f"DelegateDoctor measures Arm performance, so a non-Arm image is "
            f"never used as a benchmark target."
        )
    return abi


def start_delegate_doctor_emulator(environment: AndroidEnvironment,
                                   name: str = AVD_NAME,
                                   announce=print,
                                   sleep=time.sleep,
                                   now=time.monotonic) -> str:
    """Start the managed AVD and return the adb serial it came up as."""
    if not avd_exists(environment, name):
        raise EmulatorError(
            f"The AVD {name} does not exist.\n"
            f"\n"
            f"Create it with:\n"
            f"\n"
            f"    delegate-doctor setup-android"
        )

    before = _adb_serials(environment)
    announce(f"Starting {name}...")
    launch_emulator_process(environment, name)

    serial = wait_for_new_serial(environment, before, sleep=sleep, now=now)
    wait_for_boot(environment, serial, sleep=sleep, now=now)
    verify_abi(environment, serial)
    announce(f"Emulator ready              {serial} · {SYSTEM_IMAGE_ABI}")
    return serial
