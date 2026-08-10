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


def run_adb(*args: str, check: bool = True) -> str:
    """Run an adb command and return its stdout."""
    try:
        completed = subprocess.run(
            ["adb", *args], capture_output=True, text=True, check=check
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
            "`doctor` runs the model on a real Arm64 target, so it needs a device\n"
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

    info = DeviceInfo(
        model=run_adb("shell", "getprop", "ro.product.model").strip(),
        abi=run_adb("shell", "getprop", "ro.product.cpu.abi").strip(),
        android_release=run_adb("shell", "getprop", "ro.build.version.release").strip(),
        sdk_level=run_adb("shell", "getprop", "ro.build.version.sdk").strip(),
        hardware=run_adb("shell", "getprop", "ro.hardware").strip(),
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


def push_file(local_path: str, remote_name: str | None = None) -> str:
    """Copy a file to the device work directory and return its remote path."""
    remote_path = f"{DEVICE_WORK_DIR}/{remote_name or os.path.basename(local_path)}"
    run_adb("push", local_path, remote_path)
    return remote_path


def prepare_work_dir() -> None:
    run_adb("shell", "mkdir", "-p", DEVICE_WORK_DIR)


def push_runner(local_runner_path: str) -> str:
    """Push a runner binary and make it executable on the device."""
    remote_path = push_file(local_runner_path)
    run_adb("shell", "chmod", "+x", remote_path)
    return remote_path


def run_on_device(shell_command: str) -> None:
    """Run a shell command on the device, raising on a non-zero exit."""
    completed = subprocess.run(
        ["adb", "shell", f"{shell_command}; echo DD_EXIT=$?"],
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


def read_executorch_logcat() -> str:
    """Return the ExecuTorch log lines currently in logcat.

    The Android build of ExecuTorch logs through the Android logging system
    rather than stdout, so the runner's per-iteration timings have to be read
    back from logcat rather than captured from the shell command.
    """
    return run_adb("logcat", "-d")


def clear_logcat() -> None:
    run_adb("logcat", "-c")
