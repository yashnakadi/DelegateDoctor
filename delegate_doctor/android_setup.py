"""Build the Android Arm64 runners DelegateDoctor needs, reproducibly.

`delegate-doctor setup-android` runs everything in this file. It downloads the
exact ExecuTorch source revision that matches the supported ExecuTorch Python
package, cross-compiles two `arm64-v8a` runners, and installs them into
`runners/`.

Why the source is needed at all
-------------------------------
The ExecuTorch *Python* package gives us export, partitioning and the Inspector.
It does not ship a native Android binary that can execute a .pte on a phone.
That binary - ExecuTorch's own `executor_runner` - has to be cross-compiled, and
cross-compiling it needs the ExecuTorch C++ source tree.

The source is build-time only. Once the two runners are in `runners/`, normal
`delegate-doctor doctor` runs never touch `.build/` again.

Why two runners
---------------
  * profiling runner  - event tracer ON, writes an ETDump trace. Used to work
    out where time goes (runtime-weighted delegation, hotspot ranking).
  * benchmark runner  - event tracer OFF. Used for latency measurement.

They are kept separate on purpose: the tracer adds per-instruction overhead, so
an instrumented binary must never produce the numbers a repair decision rests
on. Merging them would quietly corrupt every benchmark.

Reproducibility
---------------
The commit below is pinned. This prototype has been validated against exactly
one ExecuTorch version, and building a native runtime from a different revision
than the Python package would be a silent correctness hazard, so setup refuses
to run against anything else.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from . import device

# ---------------------------------------------------------------------------
# Pinned versions. The single obvious place to change if this is ever updated.
# ---------------------------------------------------------------------------

SUPPORTED_EXECUTORCH_VERSION = "1.4.0"
EXECUTORCH_COMMIT = "3dd7ccd1d863fad22639dd2d918ae34a41ce45f0"
EXECUTORCH_REPOSITORY = "https://github.com/pytorch/executorch.git"

# Android settings, matching what the prototype was validated with.
ANDROID_ABI = "arm64-v8a"
ANDROID_PLATFORM = "android-28"

# ExecuTorch's CMake refuses to build unless its source directory is literally
# named "executorch", so the checkout directory name is not a free choice.
EXECUTORCH_DIR_NAME = "executorch"


class SetupError(RuntimeError):
    """An expected setup problem, reported as a short message rather than a traceback."""


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------


def check_command(name: str, install_hint: str = "") -> str:
    """Confirm an external tool is on PATH and return its full path."""
    path = shutil.which(name)
    if path is None:
        message = f"ERROR: {name} is not installed."
        if install_hint:
            message += f"\n\n{install_hint}"
        raise SetupError(message)
    return path


def get_installed_executorch_version() -> str:
    """Read the version of the installed ExecuTorch Python package."""
    try:
        from executorch.version import __version__ as installed_version
    except Exception:
        # Fall back to package metadata in case the layout changes.
        try:
            from importlib.metadata import version

            installed_version = version("executorch")
        except Exception:
            raise SetupError(
                "ERROR: the ExecuTorch Python package is not installed.\n\n"
                "Install the project's dependencies first:\n"
                "    pip install -e ."
            )
    return str(installed_version)


def check_executorch_version() -> str:
    """Refuse to build a native runtime that does not match the Python package."""
    installed_version = get_installed_executorch_version()
    if installed_version != SUPPORTED_EXECUTORCH_VERSION:
        raise SetupError(
            f"Installed ExecuTorch: {installed_version}\n"
            f"Supported ExecuTorch: {SUPPORTED_EXECUTORCH_VERSION}\n"
            f"\n"
            f"ERROR:\n"
            f"This DelegateDoctor prototype has only been validated with\n"
            f"ExecuTorch {SUPPORTED_EXECUTORCH_VERSION}.\n"
            f"\n"
            f"Building native runners from a different source revision than the\n"
            f"installed Python package can produce wrong results rather than an\n"
            f"obvious failure, so setup stops here.\n"
            f"\n"
            f"Install the supported environment and try again:\n"
            f"    pip install 'executorch=={SUPPORTED_EXECUTORCH_VERSION}'"
        )
    return installed_version


# Environment variables that conventionally point at an NDK, most specific first.
NDK_ENVIRONMENT_VARIABLES = ("ANDROID_NDK_HOME", "ANDROID_NDK_ROOT", "ANDROID_NDK")

# Environment variables that point at an SDK containing an `ndk/<version>` dir.
SDK_ENVIRONMENT_VARIABLES = ("ANDROID_HOME", "ANDROID_SDK_ROOT")

# Last-resort locations, for the common default installs.
DEFAULT_SDK_LOCATIONS = (
    "~/Library/Android/sdk",                      # Android Studio on macOS
    "~/Android/Sdk",                              # Android Studio on Linux
    "/opt/homebrew/share/android-commandlinetools",  # Homebrew cask
    "/usr/local/share/android-commandlinetools",
)

NDK_NOT_FOUND_MESSAGE = """ERROR: Android NDK not found.

Set ANDROID_NDK_HOME to your Android NDK directory.

Example:
export ANDROID_NDK_HOME="$HOME/Library/Android/sdk/ndk/27.2.12479018"

If you do not have an NDK yet, install one with the Android SDK manager:
    sdkmanager --install "ndk;27.2.12479018"
"""


def _is_ndk_directory(path: Path) -> bool:
    """An NDK is identified by the CMake toolchain file DelegateDoctor needs."""
    return (path / "build" / "cmake" / "android.toolchain.cmake").is_file()


def _newest_ndk_in_sdk(sdk_dir: Path) -> Path | None:
    """Pick the highest-numbered NDK inside an SDK directory, if any."""
    ndk_parent = sdk_dir / "ndk"
    if not ndk_parent.is_dir():
        return None
    candidates = [child for child in ndk_parent.iterdir() if _is_ndk_directory(child)]
    if not candidates:
        return None
    # Directory names are version numbers like 27.2.12479018; sorting the
    # numeric parts gives a sensible "newest" without a version library.
    def version_key(path: Path):
        parts = []
        for piece in path.name.split("."):
            parts.append(int(piece) if piece.isdigit() else 0)
        return parts

    return sorted(candidates, key=version_key)[-1]


def find_android_ndk() -> Path:
    """Locate an Android NDK, preferring explicit environment variables."""
    for variable in NDK_ENVIRONMENT_VARIABLES:
        value = os.environ.get(variable)
        if value:
            candidate = Path(value).expanduser()
            if _is_ndk_directory(candidate):
                return candidate
            raise SetupError(
                f"ERROR: {variable} is set to {candidate}, but that does not look "
                f"like an Android NDK.\n\n"
                f"Expected to find:\n"
                f"    {candidate / 'build' / 'cmake' / 'android.toolchain.cmake'}"
            )

    search_locations = []
    for variable in SDK_ENVIRONMENT_VARIABLES:
        value = os.environ.get(variable)
        if value:
            search_locations.append(Path(value).expanduser())
    for location in DEFAULT_SDK_LOCATIONS:
        search_locations.append(Path(location).expanduser())

    for sdk_dir in search_locations:
        if not sdk_dir.is_dir():
            continue
        ndk_dir = _newest_ndk_in_sdk(sdk_dir)
        if ndk_dir is not None:
            return ndk_dir

    raise SetupError(NDK_NOT_FOUND_MESSAGE)


# ---------------------------------------------------------------------------
# Running build steps
# ---------------------------------------------------------------------------


def run_step(
    description: str,
    command: list,
    log_path: Path,
    working_dir: Path | None = None,
) -> None:
    """Run one build command, sending all of its output to a log file.

    Build output is enormous and almost never interesting when things work, so
    it goes to a log. On failure we print the log path rather than thousands of
    lines of compiler output.
    """
    print(f"  {description}...")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log_file:
        log_file.write(f"\n\n=== {description} ===\n")
        log_file.write(f"$ {' '.join(str(part) for part in command)}\n\n")
        log_file.flush()
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=str(working_dir) if working_dir else None,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        raise SetupError(
            f"ERROR: {description} failed.\n"
            f"See:\n"
            f"{log_path}"
        )


# ---------------------------------------------------------------------------
# ExecuTorch source checkout
# ---------------------------------------------------------------------------


def prepare_build_directory(project_root: Path) -> Path:
    """Create the project-local, git-ignored build workspace."""
    build_dir = project_root / ".build"
    build_dir.mkdir(exist_ok=True)
    (build_dir / "logs").mkdir(exist_ok=True)
    return build_dir


def current_commit(source_dir: Path) -> str | None:
    """The commit currently checked out, or None if this is not a git checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(source_dir),
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def looks_like_executorch_checkout(source_dir: Path) -> bool:
    """Cheap sanity check that a directory really is an ExecuTorch tree."""
    return (source_dir / "CMakeLists.txt").is_file() and (
        source_dir / "backends" / "xnnpack"
    ).is_dir()


def fetch_executorch_source(source_dir: Path, log_path: Path) -> None:
    """Fetch the pinned ExecuTorch commit and its submodules into `source_dir`.

    A shallow fetch of one commit is used rather than a full clone: ExecuTorch
    plus submodules is a large repository and only this revision is wanted.
    """
    source_dir.mkdir(parents=True, exist_ok=True)

    if not (source_dir / ".git").is_dir():
        run_step("initializing repository", ["git", "init", "-q", "."],
                 log_path, working_dir=source_dir)
        run_step(
            "adding remote",
            ["git", "remote", "add", "origin", EXECUTORCH_REPOSITORY],
            log_path,
            working_dir=source_dir,
        )

    run_step(
        f"fetching ExecuTorch {EXECUTORCH_COMMIT[:12]} (this downloads a large "
        f"repository and needs internet access)",
        ["git", "fetch", "--depth", "1", "origin", EXECUTORCH_COMMIT],
        log_path,
        working_dir=source_dir,
    )
    run_step(
        "checking out pinned commit",
        ["git", "checkout", "-q", "--force", "FETCH_HEAD"],
        log_path,
        working_dir=source_dir,
    )
    run_step(
        "fetching submodules (large; needs internet access)",
        ["git", "submodule", "update", "--init", "--recursive", "--depth", "1"],
        log_path,
        working_dir=source_dir,
    )


def prepare_executorch_source(build_dir: Path) -> Path:
    """Make sure `.build/executorch` exists at exactly the pinned commit.

    This directory belongs to DelegateDoctor's build process and is treated as
    disposable. A sibling ExecuTorch checkout elsewhere on the machine is never
    read or modified.
    """
    source_dir = build_dir / EXECUTORCH_DIR_NAME
    log_path = build_dir / "logs" / "source_checkout.log"

    if source_dir.is_dir() and looks_like_executorch_checkout(source_dir):
        if current_commit(source_dir) == EXECUTORCH_COMMIT:
            print(f"  ExecuTorch source already at {EXECUTORCH_COMMIT[:12]}, reusing it.")
            return source_dir
        print("  Cached ExecuTorch source is at the wrong commit, re-checking out.")

    fetch_executorch_source(source_dir, log_path)

    checked_out = current_commit(source_dir)
    if checked_out != EXECUTORCH_COMMIT:
        raise SetupError(
            f"ERROR: ExecuTorch source is at {checked_out}, expected "
            f"{EXECUTORCH_COMMIT}.\n"
            f"Delete {source_dir} and run setup again."
        )
    return source_dir


# ---------------------------------------------------------------------------
# Runner builds
# ---------------------------------------------------------------------------


def cmake_configure_command(
    source_dir: Path,
    build_output_dir: Path,
    ndk_dir: Path,
    event_tracer: bool,
) -> list:
    """The proven configure command, one flag different between the two runners.

    This mirrors the CMake invocation the prototype was validated with. The
    android preset keeps XNNPACK, optimized kernels and quantized kernels on,
    matching the configuration ExecuTorch ships in its Android AAR. The LLM and
    training extensions are switched off purely to shorten the build; they are
    not used by DelegateDoctor.
    """
    toolchain_file = ndk_dir / "build" / "cmake" / "android.toolchain.cmake"
    return [
        "cmake",
        "-S", str(source_dir),
        "-B", str(build_output_dir),
        f"-DEXECUTORCH_BUILD_PRESET_FILE={source_dir / 'tools' / 'cmake' / 'preset' / 'android.cmake'}",
        f"-DANDROID_ABI={ANDROID_ABI}",
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file}",
        f"-DANDROID_PLATFORM={ANDROID_PLATFORM}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON",
        "-DEXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL=ON",
        f"-DEXECUTORCH_ENABLE_EVENT_TRACER={'ON' if event_tracer else 'OFF'}",
        "-DEXECUTORCH_BUILD_ANDROID_JNI=OFF",
        "-DEXECUTORCH_BUILD_EXTENSION_LLM=OFF",
        "-DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=OFF",
        "-DEXECUTORCH_BUILD_KERNELS_LLM=OFF",
        "-DEXECUTORCH_BUILD_EXTENSION_TRAINING=OFF",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
    ]


def build_runner(
    source_dir: Path,
    build_dir: Path,
    ndk_dir: Path,
    event_tracer: bool,
    label: str,
    parallel_jobs: int,
) -> Path:
    """Configure and build one `executor_runner`, returning the built binary."""
    build_output_dir = build_dir / f"cmake-out-android-{label}"
    log_path = build_dir / "logs" / f"{label}_runner_build.log"
    if log_path.exists():
        log_path.unlink()

    print(f"\nBuilding {label} runner (event tracer "
          f"{'ON' if event_tracer else 'OFF'})")

    run_step(
        f"configuring {label} runner",
        cmake_configure_command(source_dir, build_output_dir, ndk_dir, event_tracer),
        log_path,
    )
    run_step(
        f"compiling {label} runner (this takes several minutes)",
        ["cmake", "--build", str(build_output_dir), "-j", str(parallel_jobs)],
        log_path,
    )

    built_binary = build_output_dir / "executor_runner"
    if not built_binary.is_file():
        raise SetupError(
            f"ERROR: the {label} runner build finished but produced no binary at\n"
            f"{built_binary}\n"
            f"See:\n{log_path}"
        )
    return built_binary


def find_llvm_strip(ndk_dir: Path) -> Path | None:
    """Locate llvm-strip inside the NDK, if it is there.

    Stripping is optional. An unstripped runner is around 150 MB because of
    debug info; stripped it is about 7 MB. Nothing functional depends on it.
    """
    toolchains = ndk_dir / "toolchains" / "llvm" / "prebuilt"
    if not toolchains.is_dir():
        return None
    for host_dir in sorted(toolchains.iterdir()):
        candidate = host_dir / "bin" / "llvm-strip"
        if candidate.is_file():
            return candidate
    return None


def install_runner(
    built_binary: Path,
    destination: Path,
    llvm_strip: Path | None,
    log_path: Path,
) -> None:
    """Put a built runner into `runners/`, stripped when possible."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    if llvm_strip is not None:
        run_step(
            f"stripping and installing {destination.name}",
            [llvm_strip, built_binary, "-o", destination],
            log_path,
        )
    else:
        print(f"  installing {destination.name} (llvm-strip not found, keeping "
              f"debug info)...")
        shutil.copy2(built_binary, destination)

    # Make it executable for everyone who can read it.
    current_mode = destination.stat().st_mode
    destination.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def describe_binary(path: Path) -> str:
    """Ask the `file` tool what a binary is, returning '' if unavailable."""
    if shutil.which("file") is None:
        return ""
    completed = subprocess.run(
        ["file", "-b", str(path)], capture_output=True, text=True
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def verify_runner(path: Path) -> str:
    """Check an installed runner looks usable. Returns a short description.

    No Android device is needed for this. Architecture checking depends on the
    host having the `file` tool; when it is missing we say so rather than
    implying the binary was verified.
    """
    if not path.is_file():
        raise SetupError(f"ERROR: expected runner is missing: {path}")
    if path.stat().st_size == 0:
        raise SetupError(f"ERROR: runner is empty: {path}")
    if not os.access(path, os.X_OK):
        raise SetupError(f"ERROR: runner is not executable: {path}")

    description = describe_binary(path)
    if not description:
        return "architecture not checked (`file` tool unavailable on this host)"

    lowered = description.lower()
    if "aarch64" not in lowered and "arm64" not in lowered:
        raise SetupError(
            f"ERROR: {path.name} does not look like an Arm64 binary.\n"
            f"`file` reports: {description}\n"
            f"Delete the .build directory and run setup again."
        )
    if "elf" not in lowered:
        raise SetupError(
            f"ERROR: {path.name} is not an ELF executable, so it cannot run on "
            f"Android.\n`file` reports: {description}"
        )
    return description


def runners_already_installed(runners_dir: Path) -> bool:
    """Are both runners present and passing verification?"""
    for runner_name in (device.ETDUMP_RUNNER_NAME, device.BENCH_RUNNER_NAME):
        path = runners_dir / runner_name
        try:
            verify_runner(path)
        except SetupError:
            return False
    return True


# ---------------------------------------------------------------------------
# The whole workflow
# ---------------------------------------------------------------------------


def setup_android_runners(
    project_root: Path,
    runners_dir: Path,
    rebuild: bool = False,
    parallel_jobs: int = 10,
) -> int:
    """Build and install both Android runners. Returns a process exit code."""
    print("DelegateDoctor Android setup\n")

    # 1. Environment. All checks happen before anything is downloaded or built,
    #    so a missing tool fails in seconds rather than after a long clone.
    installed_version = check_executorch_version()
    print(f"Installed ExecuTorch: {installed_version} (supported)")

    check_command("git", "Install git and try again.")
    check_command(
        "cmake",
        "Install CMake and try again, e.g.\n    brew install cmake",
    )
    ndk_dir = find_android_ndk()
    print(f"Android NDK:          {ndk_dir}")
    print(f"Python:               {sys.executable}")
    print(f"Target ABI:           {ANDROID_ABI}")
    print(f"Source commit:        {EXECUTORCH_COMMIT}")

    # 2. Skip the whole build if there is nothing to do.
    if not rebuild and runners_already_installed(runners_dir):
        print("\nProfiling runner already exists.")
        print("Benchmark runner already exists.")
        print("\nAndroid runner setup is complete.")
        print(f"Runners: {runners_dir}")
        print("\nRe-run with --rebuild to build them again.")
        return 0

    # 3. Source.
    build_dir = prepare_build_directory(project_root)
    print(f"\nBuild workspace: {build_dir}")
    print("Preparing ExecuTorch source")
    source_dir = prepare_executorch_source(build_dir)

    # 4. Two builds, differing only in the event tracer.
    profiling_binary = build_runner(
        source_dir, build_dir, ndk_dir,
        event_tracer=True, label="etdump", parallel_jobs=parallel_jobs,
    )
    benchmark_binary = build_runner(
        source_dir, build_dir, ndk_dir,
        event_tracer=False, label="bench", parallel_jobs=parallel_jobs,
    )

    # 5. Install and verify.
    print("\nInstalling runners")
    llvm_strip = find_llvm_strip(ndk_dir)
    install_log = build_dir / "logs" / "install.log"

    profiling_destination = runners_dir / device.ETDUMP_RUNNER_NAME
    benchmark_destination = runners_dir / device.BENCH_RUNNER_NAME
    install_runner(profiling_binary, profiling_destination, llvm_strip, install_log)
    install_runner(benchmark_binary, benchmark_destination, llvm_strip, install_log)

    profiling_description = verify_runner(profiling_destination)
    benchmark_description = verify_runner(benchmark_destination)

    print("\nAndroid runner setup is complete.\n")
    print(f"Profiling runner (ETDump, tracer ON):")
    print(f"  {profiling_destination}")
    print(f"  {profiling_description}")
    print(f"Benchmark runner (tracer OFF):")
    print(f"  {benchmark_destination}")
    print(f"  {benchmark_description}")
    print(f"\nExecuTorch source cached in {source_dir}")
    print("It is only needed to rebuild the runners; analysis does not read it.")
    print("\nNext:\n"
          "    python examples/<model>.py                          # a demo\n"
          "    delegate-doctor optimize model.pt2 --inputs inputs.pt")
    return 0
