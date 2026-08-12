"""Talking to the Arm64 Android target over adb.

Both profiling and benchmarking need to push files and run a binary on the
device, so that logic lives here once.

Two separate runner binaries are used on purpose:

  * `executor_runner_etdump`  - built with the ExecuTorch event tracer ON.
     Emits a per-operator ETDump trace. Used for profiling only, because the
     tracer adds measurable per-instruction overhead.

  * `executor_runner_bench`   - built with the event tracer OFF.
     Used for latency benchmarking, so instrumentation cannot distort results.

Both are the stock ExecuTorch `executor_runner`, cross-compiled for arm64-v8a.
See the README for the exact CMake commands.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

# Where we stage files on the device. /data/local/tmp is world-writable and
# executable on a normal (non-rooted) Android device.
DEVICE_WORK_DIR = "/data/local/tmp/delegate_doctor"

BENCH_RUNNER_NAME = "executor_runner_bench"
ETDUMP_RUNNER_NAME = "executor_runner_etdump"


class DeviceError(RuntimeError):
    """Raised when the Arm target or its tooling is not usable."""


@dataclass
class DeviceInfo:
    # adb serial of the target we selected. Every later adb call passes this
    # explicitly with `adb -s <serial>`, so profiling, verification and
    # benchmarking can never drift onto a different device if more than one is
    # attached later in the run.
    serial: str
    model: str
    abi: str
    android_release: str
    sdk_level: str
    hardware: str

    @property
    def is_emulator(self) -> bool:
        # 'ranchu' is the Android emulator's virtual platform. Goldfish is the
        # older name and still shows up on some images.
        return self.hardware in ("ranchu", "goldfish") or "sdk" in self.model.lower()

    def describe(self) -> str:
        kind = "Arm64 Android emulator" if self.is_emulator else "Arm64 Android device"
        return f"{kind} - {self.model} ({self.abi}, Android {self.android_release})"

    def short_description(self) -> str:
        """Compact one-line form for the console header.

        Emulators stay labelled: their numbers are not handset numbers, and that
        distinction must survive being made concise.
        """
        text = f"{self.model} · {self.abi} · Android {self.android_release}"
        if self.is_emulator:
            text += " (emulator)"
        return text


def run_adb(*args: str, check: bool = True, serial: str | None = None) -> str:
    """Run an adb command and return its stdout.

    `serial` selects a specific target with `adb -s`. Passing it explicitly is
    safer than relying on adb's implicit choice, which fails or picks the wrong
    device as soon as a second target appears.
    """
    command = ["adb"]
    if serial:
        command += ["-s", serial]
    command += list(args)
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=check
        )
    except FileNotFoundError:
        raise DeviceError(
            "adb was not found on PATH. Install the Android platform tools, e.g.\n"
            "  brew install --cask android-platform-tools"
        )
    return completed.stdout


def require_device() -> DeviceInfo:
    """Check that exactly one usable Arm64 device is attached."""
    listing = run_adb("devices")
    connected = []
    for line in listing.splitlines()[1:]:
        if line.strip().endswith("device"):
            connected.append(line.split()[0])

    if not connected:
        raise DeviceError(
            "No Arm64 Android target is attached.\n"
            "\n"
            "DelegateDoctor runs the model on a real Arm64 target, so it needs a device\n"
            "or emulator visible to adb. Check with:\n"
            "\n"
            "    adb devices\n"
            "\n"
            "A physical Arm64 phone is preferred. To start an emulator instead:\n"
            "\n"
            "    $ANDROID_HOME/emulator/emulator -avd <your-arm64-avd> -no-window -no-audio -gpu off\n"
            "\n"
            "See the 'Connect an Arm64 Android target' section of the README."
        )
    if len(connected) > 1:
        raise DeviceError(
            f"Multiple devices attached ({', '.join(connected)}). "
            "Disconnect all but one, or set ANDROID_SERIAL."
        )

    serial = connected[0]
    info = DeviceInfo(
        serial=serial,
        model=run_adb("shell", "getprop", "ro.product.model", serial=serial).strip(),
        abi=run_adb("shell", "getprop", "ro.product.cpu.abi", serial=serial).strip(),
        android_release=run_adb(
            "shell", "getprop", "ro.build.version.release", serial=serial
        ).strip(),
        sdk_level=run_adb(
            "shell", "getprop", "ro.build.version.sdk", serial=serial
        ).strip(),
        hardware=run_adb("shell", "getprop", "ro.hardware", serial=serial).strip(),
    )

    if info.abi != "arm64-v8a":
        raise DeviceError(
            f"Unsupported target ABI.\n"
            f"\n"
            f"    attached: {info.abi}  ({info.model})\n"
            f"    required: arm64-v8a\n"
            f"\n"
            f"DelegateDoctor targets Arm64 only, and the runners in runners/ are\n"
            f"cross-compiled for arm64-v8a. An x86_64 emulator image will not work;\n"
            f"use an arm64-v8a system image or a physical Arm64 phone."
        )
    return info


def find_runner(runners_dir: str, runner_name: str) -> str:
    """Locate a cross-compiled runner binary and explain clearly if it is absent."""
    path = os.path.join(runners_dir, runner_name)
    if not os.path.isfile(path):
        raise DeviceError(
            "Android runners are not installed.\n"
            "\n"
            "Run:\n"
            "\n"
            "    delegate-doctor setup-android\n"
            "\n"
            "then retry.\n"
            "\n"
            f"(missing: {path})"
        )
    return path


def push_file(
    local_path: str, remote_name: str | None = None, serial: str | None = None
) -> str:
    """Copy a file to the device work directory and return its remote path."""
    remote_path = f"{DEVICE_WORK_DIR}/{remote_name or os.path.basename(local_path)}"
    run_adb("push", local_path, remote_path, serial=serial)
    return remote_path


def pull_file(remote_path: str, local_path: str, serial: str | None = None) -> str:
    """Copy a file back from the device."""
    parent = os.path.dirname(local_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    run_adb("pull", remote_path, local_path, serial=serial)
    return local_path


def remove_remote_files(pattern: str, serial: str | None = None) -> None:
    """Delete files in the device work directory. Never fails the run."""
    run_adb(
        "shell", f"rm -f {DEVICE_WORK_DIR}/{pattern}", check=False, serial=serial
    )


def prepare_work_dir(serial: str | None = None) -> None:
    run_adb("shell", "mkdir", "-p", DEVICE_WORK_DIR, serial=serial)


def push_runner(local_runner_path: str, serial: str | None = None) -> str:
    """Push a runner binary and make it executable on the device."""
    remote_path = push_file(local_runner_path, serial=serial)
    run_adb("shell", "chmod", "+x", remote_path, serial=serial)
    return remote_path


def run_on_device(shell_command: str, serial: str | None = None) -> None:
    """Run a shell command on the device, raising on a non-zero exit."""
    command = ["adb"]
    if serial:
        command += ["-s", serial]
    command += ["shell", f"{shell_command}; echo DD_EXIT=$?"]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )
    if "DD_EXIT=0" not in completed.stdout:
        raise DeviceError(
            f"Command failed on device:\n  {shell_command}\n"
            f"stdout: {completed.stdout.strip()[:500]}\n"
            f"stderr: {completed.stderr.strip()[:500]}"
        )


def read_executorch_logcat(serial: str | None = None) -> str:
    """Return the ExecuTorch log lines currently in logcat.

    The Android build of ExecuTorch logs through the Android logging system
    rather than stdout, so the runner's per-iteration timings have to be read
    back from logcat rather than captured from the shell command.
    """
    return run_adb("logcat", "-d", serial=serial)


def clear_logcat(serial: str | None = None) -> None:
    run_adb("logcat", "-c", serial=serial)
