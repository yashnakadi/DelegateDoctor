"""Host detection, SDK discovery, and the Arm-only emulator policy.

Offline and filesystem-only: no SDK, no adb, no sdkmanager. Every host case is
injected, so macOS, Windows and Linux behaviour is exercised from one machine -
which is exactly why the report at the end of this phase distinguishes "tested
under mocks" from "validated on real hardware".
"""

from pathlib import Path

import pytest

from delegate_doctor import android_environment as env
from delegate_doctor.android_environment import HostPlatform


def host(os_name=env.OS_MACOS, architecture=env.ARCH_ARM64):
    return HostPlatform(os_name=os_name, architecture=architecture,
                        raw_system=os_name, raw_machine=architecture)


# --- host detection ----------------------------------------------------------

@pytest.mark.parametrize("system, machine, expected_os, expected_arch", [
    ("Darwin", "arm64", env.OS_MACOS, env.ARCH_ARM64),
    ("Darwin", "x86_64", env.OS_MACOS, env.ARCH_X86_64),
    ("Windows", "AMD64", env.OS_WINDOWS, env.ARCH_X86_64),
    ("Linux", "x86_64", env.OS_LINUX, env.ARCH_X86_64),
    ("Linux", "aarch64", env.OS_LINUX, env.ARCH_ARM64),
    ("Windows", "ARM64", env.OS_WINDOWS, env.ARCH_ARM64),
])
def test_hosts_normalize(system, machine, expected_os, expected_arch):
    detected = env.detect_host(system=system, machine=machine)
    assert detected.os_name == expected_os
    assert detected.architecture == expected_arch


def test_an_unknown_host_is_not_guessed():
    detected = env.detect_host(system="Plan9", machine="sparc")
    assert detected.os_name == env.OS_UNKNOWN
    assert detected.architecture == env.ARCH_UNKNOWN


def test_the_raw_values_are_preserved_for_diagnostics():
    detected = env.detect_host(system="Darwin", machine="arm64")
    assert detected.raw_system == "Darwin"
    assert detected.raw_machine == "arm64"


def test_detecting_the_real_host_does_not_raise():
    assert env.detect_host().describe()


# --- the Arm-only emulator policy (the correctness requirement) --------------











# --- SDK discovery -----------------------------------------------------------

def make_sdk(root: Path, windows=False) -> Path:
    """A directory that looks like an Android SDK."""
    suffix_exe = ".exe" if windows else ""
    suffix_bat = ".bat" if windows else ""
    (root / "platform-tools").mkdir(parents=True)
    (root / "platform-tools" / f"adb{suffix_exe}").write_text("x")
    tools = root / "cmdline-tools" / "latest" / "bin"
    tools.mkdir(parents=True)
    (tools / f"sdkmanager{suffix_bat}").write_text("x")
    (tools / f"avdmanager{suffix_bat}").write_text("x")
    return root


def test_android_home_is_used(tmp_path):
    sdk = make_sdk(tmp_path / "sdk")
    assert env.find_sdk_root({"ANDROID_HOME": str(sdk)}) == sdk


def test_android_sdk_root_is_used(tmp_path):
    sdk = make_sdk(tmp_path / "sdk")
    assert env.find_sdk_root({"ANDROID_SDK_ROOT": str(sdk)}) == sdk


def test_android_home_wins_over_sdk_root(tmp_path):
    first = make_sdk(tmp_path / "first")
    second = make_sdk(tmp_path / "second")
    found = env.find_sdk_root({"ANDROID_HOME": str(first),
                               "ANDROID_SDK_ROOT": str(second)})
    assert found == first


def test_a_nonexistent_sdk_variable_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "DEFAULT_SDK_LOCATIONS", ())
    assert env.find_sdk_root({"ANDROID_HOME": str(tmp_path / "nope")}) is None


def test_no_sdk_anywhere_returns_none(monkeypatch):
    monkeypatch.setattr(env, "DEFAULT_SDK_LOCATIONS", ())
    assert env.find_sdk_root({}) is None


def test_discovery_never_walks_the_filesystem(tmp_path, monkeypatch):
    """A deep SDK must NOT be found: discovery checks fixed locations only."""
    buried = make_sdk(tmp_path / "a" / "b" / "c" / "sdk")
    monkeypatch.setattr(env, "DEFAULT_SDK_LOCATIONS", ())
    assert env.find_sdk_root({"ANDROID_HOME": str(tmp_path)}) == tmp_path
    # ...and the buried one was never consulted as a tool source.
    tool = env.find_tool("adb", host(), tmp_path, path_lookup=lambda name: None)
    assert not tool.found


# --- tool location -----------------------------------------------------------

def test_tools_are_found_inside_the_sdk(tmp_path):
    sdk = make_sdk(tmp_path / "sdk")
    for name in ("adb", "sdkmanager"):
        tool = env.find_tool(name, host(), sdk, path_lookup=lambda n: None)
        assert tool.found, name
        assert sdk in tool.path.parents


def test_windows_executable_suffixes_are_used(tmp_path):
    sdk = make_sdk(tmp_path / "sdk", windows=True)
    windows = host(env.OS_WINDOWS, env.ARCH_X86_64)

    adb = env.find_tool("adb", windows, sdk, path_lookup=lambda n: None)
    assert adb.path.name == "adb.exe"

    sdkmanager = env.find_tool("sdkmanager", windows, sdk,
                               path_lookup=lambda n: None)
    assert sdkmanager.path.name == "sdkmanager.bat"


def test_a_versioned_cmdline_tools_directory_is_handled(tmp_path):
    sdk = tmp_path / "sdk"
    versioned = sdk / "cmdline-tools" / "13.0" / "bin"
    versioned.mkdir(parents=True)
    (versioned / "sdkmanager").write_text("x")

    tool = env.find_tool("sdkmanager", host(), sdk, path_lookup=lambda n: None)
    assert tool.found
    assert tool.path.parent.parent.name == "13.0"


def test_latest_is_preferred_over_a_versioned_directory(tmp_path):
    sdk = tmp_path / "sdk"
    for name in ("latest", "13.0"):
        directory = sdk / "cmdline-tools" / name / "bin"
        directory.mkdir(parents=True)
        (directory / "sdkmanager").write_text("x")

    tool = env.find_tool("sdkmanager", host(), sdk, path_lookup=lambda n: None)
    assert tool.path.parent.parent.name == "latest"


def test_a_tool_on_path_is_used_when_the_sdk_lacks_it(tmp_path):
    tool = env.find_tool("adb", host(), None,
                         path_lookup=lambda name: "/usr/local/bin/adb")
    assert tool.found
    assert tool.path == Path("/usr/local/bin/adb")


def test_the_sdk_copy_is_preferred_over_path(tmp_path):
    """A PATH hit could belong to an unrelated install."""
    sdk = make_sdk(tmp_path / "sdk")
    tool = env.find_tool("adb", host(), sdk,
                         path_lookup=lambda name: "/somewhere/else/adb")
    assert sdk in tool.path.parents


def test_a_missing_tool_is_reported_not_guessed():
    tool = env.find_tool("adb", host(), None, path_lookup=lambda name: None)
    assert not tool.found
    assert tool.status == "MISSING"
    assert tool.path is None


# --- NDK ---------------------------------------------------------------------

def make_ndk(root: Path) -> Path:
    (root / "build" / "cmake").mkdir(parents=True)
    (root / "build" / "cmake" / "android.toolchain.cmake").write_text("x")
    return root


def test_the_ndk_environment_variable_wins(tmp_path):
    ndk = make_ndk(tmp_path / "ndk" / "27.2.12479018")
    assert env.find_ndk(None, {"ANDROID_NDK_HOME": str(ndk)}) == ndk


def test_an_sdk_ndk_is_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "DEFAULT_SDK_LOCATIONS", ())
    sdk = tmp_path / "sdk"
    make_ndk(sdk / "ndk" / "27.2.12479018")
    assert env.find_ndk(sdk, {}) is not None


def test_the_newest_ndk_is_chosen(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "DEFAULT_SDK_LOCATIONS", ())
    sdk = tmp_path / "sdk"
    for version in ("21.4.7075529", "27.2.12479018", "26.1.10909125"):
        make_ndk(sdk / "ndk" / version)
    assert env.find_ndk(sdk, {}).name == "27.2.12479018"


def test_a_directory_without_the_toolchain_is_not_an_ndk(tmp_path):
    (tmp_path / "ndk").mkdir()
    assert not env.is_ndk_directory(tmp_path / "ndk")


def test_no_ndk_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "DEFAULT_SDK_LOCATIONS", ())
    assert env.find_ndk(tmp_path, {}) is None


# --- the assembled environment ----------------------------------------------





def test_no_sdk_at_all_is_reported(monkeypatch):
    monkeypatch.setattr(env, "DEFAULT_SDK_LOCATIONS", ())
    detected = env.detect(environment={}, host=host(),
                          path_lookup=lambda name: None)
    assert not detected.has_sdk
    assert detected.tool("adb").status == "MISSING"




# --- messages ----------------------------------------------------------------

def test_the_command_line_tools_message_does_not_demand_android_studio():
    message = env.COMMAND_LINE_TOOLS_MISSING_MESSAGE
    assert "commandlinetools" in message
    assert "does not download or bundle them itself" in message
    # Android Studio may be *offered*, but must not be the only route.
    assert "cmdline-tools/latest" in message


def test_the_sdk_missing_message_says_discovery_is_not_a_search():
    assert "does not search the filesystem" in env.SDK_MISSING_MESSAGE
    assert "phone with adb also works" in env.SDK_MISSING_MESSAGE
