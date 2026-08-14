"""Tests for the Android runner setup.

None of these clone ExecuTorch, run CMake, or touch a device. Subprocess and
filesystem boundaries are mocked, so the suite stays offline and fast.
"""

import os
import stat
import subprocess
from pathlib import Path

import pytest

from delegate_doctor import android_setup, device
from delegate_doctor.android_setup import (
    EXECUTORCH_COMMIT,
    SUPPORTED_EXECUTORCH_VERSION,
    SetupError,
)


# --- version validation ----------------------------------------------------

def test_supported_executorch_version_passes(monkeypatch):
    monkeypatch.setattr(
        android_setup, "get_installed_executorch_version",
        lambda: SUPPORTED_EXECUTORCH_VERSION,
    )
    assert android_setup.check_executorch_version() == SUPPORTED_EXECUTORCH_VERSION


def test_unsupported_executorch_version_fails_clearly(monkeypatch):
    monkeypatch.setattr(
        android_setup, "get_installed_executorch_version", lambda: "1.5.0"
    )
    with pytest.raises(SetupError) as caught:
        android_setup.check_executorch_version()

    message = str(caught.value)
    assert "Installed ExecuTorch: 1.5.0" in message
    assert f"Supported ExecuTorch: {SUPPORTED_EXECUTORCH_VERSION}" in message
    assert "ERROR" in message


# --- tool validation -------------------------------------------------------

def test_missing_git_is_reported(monkeypatch):
    monkeypatch.setattr(android_setup.shutil, "which", lambda name: None)
    with pytest.raises(SetupError) as caught:
        android_setup.check_command("git")
    assert "git is not installed" in str(caught.value)


def test_missing_cmake_is_reported(monkeypatch):
    monkeypatch.setattr(android_setup.shutil, "which", lambda name: None)
    with pytest.raises(SetupError) as caught:
        android_setup.check_command("cmake", "Install CMake and try again.")
    message = str(caught.value)
    assert "cmake is not installed" in message
    assert "Install CMake" in message


def test_present_command_returns_its_path(monkeypatch):
    monkeypatch.setattr(android_setup.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert android_setup.check_command("git") == "/usr/bin/git"


# --- NDK discovery ---------------------------------------------------------

def make_fake_ndk(root: Path) -> Path:
    """Create a directory that looks like an NDK to find_android_ndk()."""
    toolchain = root / "build" / "cmake"
    toolchain.mkdir(parents=True)
    (toolchain / "android.toolchain.cmake").write_text("# fake toolchain\n")
    return root


def test_ndk_environment_variable_is_used(tmp_path, monkeypatch):
    ndk = make_fake_ndk(tmp_path / "ndk-27")
    for variable in android_setup.NDK_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ANDROID_NDK_HOME", str(ndk))

    assert android_setup.find_android_ndk() == ndk


def test_ndk_found_inside_an_sdk_directory(tmp_path, monkeypatch):
    sdk = tmp_path / "sdk"
    make_fake_ndk(sdk / "ndk" / "26.1.10909125")
    newest = make_fake_ndk(sdk / "ndk" / "27.2.12479018")

    for variable in android_setup.NDK_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ANDROID_HOME", str(sdk))
    # Keep the default locations out of it, so the test cannot pass by accident
    # on a machine that happens to have an NDK installed.
    monkeypatch.setattr(android_setup, "DEFAULT_SDK_LOCATIONS", ())

    assert android_setup.find_android_ndk() == newest


def test_missing_ndk_gives_a_useful_message(tmp_path, monkeypatch):
    for variable in android_setup.NDK_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    for variable in android_setup.SDK_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(android_setup, "DEFAULT_SDK_LOCATIONS", (str(tmp_path / "nope"),))

    with pytest.raises(SetupError) as caught:
        android_setup.find_android_ndk()

    message = str(caught.value)
    assert "Android NDK not found" in message
    assert "ANDROID_NDK_HOME" in message


def test_ndk_variable_pointing_somewhere_wrong_is_rejected(tmp_path, monkeypatch):
    empty = tmp_path / "not-an-ndk"
    empty.mkdir()
    monkeypatch.setenv("ANDROID_NDK_HOME", str(empty))

    with pytest.raises(SetupError) as caught:
        android_setup.find_android_ndk()
    assert "does not look like an Android NDK" in str(caught.value)


# --- source preparation ----------------------------------------------------

def make_fake_checkout(root: Path) -> Path:
    """A directory that passes looks_like_executorch_checkout()."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "CMakeLists.txt").write_text("# fake\n")
    (root / "backends" / "xnnpack").mkdir(parents=True)
    (root / ".git").mkdir()
    return root


def test_correct_cached_checkout_is_reused(tmp_path, monkeypatch):
    build_dir = tmp_path / ".build"
    (build_dir / "logs").mkdir(parents=True)
    make_fake_checkout(build_dir / "executorch")

    monkeypatch.setattr(android_setup, "current_commit", lambda path: EXECUTORCH_COMMIT)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("a correct cached checkout must not be re-fetched")

    monkeypatch.setattr(android_setup, "fetch_executorch_source", fail_if_called)

    source_dir = android_setup.prepare_executorch_source(build_dir)
    assert source_dir == build_dir / "executorch"


def test_missing_checkout_triggers_a_fetch(tmp_path, monkeypatch):
    build_dir = tmp_path / ".build"
    (build_dir / "logs").mkdir(parents=True)

    fetched = []

    def record_fetch(source_dir, log_path):
        fetched.append(source_dir)
        make_fake_checkout(source_dir)

    monkeypatch.setattr(android_setup, "fetch_executorch_source", record_fetch)
    monkeypatch.setattr(android_setup, "current_commit", lambda path: EXECUTORCH_COMMIT)

    android_setup.prepare_executorch_source(build_dir)
    assert fetched == [build_dir / "executorch"]


def test_wrong_commit_triggers_a_re_checkout(tmp_path, monkeypatch):
    build_dir = tmp_path / ".build"
    (build_dir / "logs").mkdir(parents=True)
    make_fake_checkout(build_dir / "executorch")

    commits = ["0000000000000000000000000000000000000000", EXECUTORCH_COMMIT]
    monkeypatch.setattr(android_setup, "current_commit", lambda path: commits.pop(0))

    fetched = []
    monkeypatch.setattr(
        android_setup, "fetch_executorch_source",
        lambda source_dir, log_path: fetched.append(source_dir),
    )

    android_setup.prepare_executorch_source(build_dir)
    assert fetched == [build_dir / "executorch"]


def test_checkout_landing_on_the_wrong_commit_is_an_error(tmp_path, monkeypatch):
    build_dir = tmp_path / ".build"
    (build_dir / "logs").mkdir(parents=True)

    monkeypatch.setattr(
        android_setup, "fetch_executorch_source",
        lambda source_dir, log_path: make_fake_checkout(source_dir),
    )
    monkeypatch.setattr(android_setup, "current_commit", lambda path: "deadbeef")

    with pytest.raises(SetupError) as caught:
        android_setup.prepare_executorch_source(build_dir)
    assert "expected" in str(caught.value)


def test_fetch_uses_the_pinned_commit_and_shallow_submodules(tmp_path, monkeypatch):
    """The exact git commands matter for reproducibility, so assert on them."""
    source_dir = tmp_path / "executorch"
    commands = []

    def record_step(description, command, log_path, working_dir=None):
        commands.append([str(part) for part in command])

    monkeypatch.setattr(android_setup, "run_step", record_step)
    android_setup.fetch_executorch_source(source_dir, tmp_path / "log.txt")

    flattened = [" ".join(command) for command in commands]
    assert any(android_setup.EXECUTORCH_REPOSITORY in line for line in flattened)
    assert f"git fetch --depth 1 origin {EXECUTORCH_COMMIT}" in flattened
    assert "git checkout -q --force FETCH_HEAD" in flattened
    assert "git submodule update --init --recursive --depth 1" in flattened


# --- build configuration ---------------------------------------------------

def test_the_two_runners_differ_only_in_the_event_tracer(tmp_path):
    ndk = make_fake_ndk(tmp_path / "ndk")
    source = tmp_path / "executorch"

    profiling = android_setup.cmake_configure_command(
        source, tmp_path / "out-etdump", ndk, event_tracer=True
    )
    benchmark = android_setup.cmake_configure_command(
        source, tmp_path / "out-bench", ndk, event_tracer=False
    )

    assert "-DEXECUTORCH_ENABLE_EVENT_TRACER=ON" in profiling
    assert "-DEXECUTORCH_ENABLE_EVENT_TRACER=OFF" in benchmark

    # Ignore the tracer flag and the build directory; everything else must match.
    def normalize(command):
        return [
            part for part in command
            if "EVENT_TRACER" not in part and "out-" not in part and part != "-B"
        ]

    assert normalize(profiling) == normalize(benchmark)


def test_build_configuration_targets_arm64_and_enables_the_runner(tmp_path):
    ndk = make_fake_ndk(tmp_path / "ndk")
    command = android_setup.cmake_configure_command(
        tmp_path / "executorch", tmp_path / "out", ndk, event_tracer=False
    )
    assert f"-DANDROID_ABI={android_setup.ANDROID_ABI}" in command
    assert "-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON" in command
    assert "-DCMAKE_BUILD_TYPE=Release" in command
    assert any("android.toolchain.cmake" in part for part in command)


# --- runner installation ---------------------------------------------------

def test_install_runner_copies_and_sets_the_executable_bit(tmp_path):
    built = tmp_path / "executor_runner"
    built.write_bytes(b"fake binary")
    destination = tmp_path / "runners" / device.BENCH_RUNNER_NAME

    android_setup.install_runner(built, destination, llvm_strip=None,
                                 log_path=tmp_path / "log.txt")

    assert destination.is_file()
    assert destination.read_bytes() == b"fake binary"
    assert os.access(destination, os.X_OK)


def test_install_runner_uses_llvm_strip_when_available(tmp_path, monkeypatch):
    built = tmp_path / "executor_runner"
    built.write_bytes(b"fake binary with debug info")
    destination = tmp_path / "runners" / device.ETDUMP_RUNNER_NAME
    strip_tool = tmp_path / "llvm-strip"

    def fake_strip(description, command, log_path, working_dir=None):
        assert str(command[0]) == str(strip_tool)
        Path(command[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(command[-1]).write_bytes(b"stripped")

    monkeypatch.setattr(android_setup, "run_step", fake_strip)
    android_setup.install_runner(built, destination, strip_tool, tmp_path / "log.txt")

    assert destination.read_bytes() == b"stripped"
    assert os.access(destination, os.X_OK)


def test_missing_build_artifact_is_a_clear_failure(tmp_path, monkeypatch):
    """A build that "succeeds" but produces nothing must still fail."""
    monkeypatch.setattr(
        android_setup, "run_step",
        lambda description, command, log_path, working_dir=None: None,
    )
    ndk = make_fake_ndk(tmp_path / "ndk")

    with pytest.raises(SetupError) as caught:
        android_setup.build_runner(
            tmp_path / "executorch", tmp_path / ".build", ndk,
            event_tracer=False, label="bench", parallel_jobs=1,
        )
    assert "produced no binary" in str(caught.value)


# --- verification ----------------------------------------------------------

def make_installed_runner(path: Path, contents: bytes = b"binary") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_verify_runner_rejects_a_missing_file(tmp_path):
    with pytest.raises(SetupError) as caught:
        android_setup.verify_runner(tmp_path / "nope")
    assert "missing" in str(caught.value)


def test_verify_runner_rejects_an_empty_file(tmp_path):
    empty = make_installed_runner(tmp_path / "runner", contents=b"")
    with pytest.raises(SetupError) as caught:
        android_setup.verify_runner(empty)
    assert "empty" in str(caught.value)


def test_verify_runner_rejects_a_non_executable_file(tmp_path):
    path = tmp_path / "runner"
    path.write_bytes(b"binary")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    with pytest.raises(SetupError) as caught:
        android_setup.verify_runner(path)
    assert "not executable" in str(caught.value)


def test_verify_runner_rejects_a_host_binary(tmp_path, monkeypatch):
    """A runner accidentally built for the host must not be reported as fine."""
    path = make_installed_runner(tmp_path / "runner")
    monkeypatch.setattr(
        android_setup, "describe_binary",
        lambda p: "Mach-O 64-bit executable arm64",
    )
    with pytest.raises(SetupError) as caught:
        android_setup.verify_runner(path)
    assert "not an ELF executable" in str(caught.value)


def test_verify_runner_accepts_an_android_arm64_binary(tmp_path, monkeypatch):
    path = make_installed_runner(tmp_path / "runner")
    monkeypatch.setattr(
        android_setup, "describe_binary",
        lambda p: "ELF 64-bit LSB pie executable, ARM aarch64, version 1 (SYSV)",
    )
    assert "aarch64" in android_setup.verify_runner(path)


def test_verify_runner_is_honest_when_it_cannot_check_architecture(tmp_path, monkeypatch):
    path = make_installed_runner(tmp_path / "runner")
    monkeypatch.setattr(android_setup, "describe_binary", lambda p: "")
    assert "not checked" in android_setup.verify_runner(path)


# --- idempotence -----------------------------------------------------------

def test_setup_reports_completion_when_runners_already_exist(tmp_path, monkeypatch, capsys):
    runners_dir = tmp_path / "runners"
    make_installed_runner(runners_dir / device.ETDUMP_RUNNER_NAME)
    make_installed_runner(runners_dir / device.BENCH_RUNNER_NAME)

    monkeypatch.setattr(
        android_setup, "get_installed_executorch_version",
        lambda: SUPPORTED_EXECUTORCH_VERSION,
    )
    monkeypatch.setattr(android_setup.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(android_setup, "find_android_ndk", lambda: tmp_path / "ndk")
    monkeypatch.setattr(
        android_setup, "describe_binary",
        lambda p: "ELF 64-bit LSB pie executable, ARM aarch64",
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("setup must not build when runners already exist")

    monkeypatch.setattr(android_setup, "prepare_executorch_source", fail_if_called)
    monkeypatch.setattr(android_setup, "build_runner", fail_if_called)

    exit_code = android_setup.setup_android_runners(tmp_path, runners_dir)

    assert exit_code == 0
    output = capsys.readouterr().out
    # Idempotence: both runners are reported READY and nothing was rebuilt.
    assert "ETDump runner          READY" in output
    assert "Benchmark runner       READY" in output
    assert "Managed Arm64 environment" in output
    assert "--rebuild" in output


def test_runners_already_installed_is_false_when_one_is_missing(tmp_path, monkeypatch):
    runners_dir = tmp_path / "runners"
    make_installed_runner(runners_dir / device.ETDUMP_RUNNER_NAME)
    monkeypatch.setattr(
        android_setup, "describe_binary", lambda p: "ELF 64-bit ARM aarch64"
    )
    assert not android_setup.runners_already_installed(runners_dir)


# --- CLI -------------------------------------------------------------------

def test_cli_dispatches_setup_android(monkeypatch):
    from delegate_doctor import cli

    calls = []

    def record(project_root, runners_dir, rebuild, parallel_jobs, **options):
        calls.append({"rebuild": rebuild, "jobs": parallel_jobs})
        return 0

    monkeypatch.setattr(cli.android_setup, "setup_android_runners", record)

    assert cli.main(["setup-android"]) == 0
    assert calls == [{"rebuild": False, "jobs": 10}]

    assert cli.main(["setup-android", "--rebuild", "--jobs", "4"]) == 0
    assert calls[-1] == {"rebuild": True, "jobs": 4}


def test_cli_reports_setup_errors_without_a_traceback(monkeypatch, capsys):
    from delegate_doctor import cli

    def raise_setup_error(**kwargs):
        raise android_setup.SetupError("ERROR: git is not installed.")

    monkeypatch.setattr(cli.android_setup, "setup_android_runners", raise_setup_error)

    assert cli.main(["setup-android"]) == 2
    assert "git is not installed" in capsys.readouterr().err


def test_a_missing_runner_points_at_setup_android(tmp_path):
    """The missing-runner message must name the command that fixes it."""
    with pytest.raises(device.DeviceError) as caught:
        device.find_runner(str(tmp_path), device.BENCH_RUNNER_NAME)

    message = str(caught.value)
    assert "Android runners are not installed" in message
    assert "delegate-doctor setup-android" in message
