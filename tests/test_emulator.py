"""Provisioning and starting the managed AVD, entirely under mocks.

`run_tool` is the single choke point for every Android command, so replacing it
gives full coverage of package inspection, AVD creation and boot handling
without an SDK, an emulator or a second of waiting.

Several tests here exist to prove a *negative*: that DelegateDoctor never
deletes an AVD, never wipes data, never touches a user's own emulators, and
never accepts a non-Arm image as a benchmark target.
"""

import subprocess

import pytest

from delegate_doctor import android_environment as env
from delegate_doctor import emulator
from delegate_doctor.android_environment import AndroidEnvironment, AndroidTool
from delegate_doctor.emulator import EmulatorError, ToolResult
# Captured at import time: conftest deliberately blocks the module attribute.
from delegate_doctor.emulator import launch_emulator_process as REAL_LAUNCH


def make_environment(tmp_path, **overrides):
    """An environment whose tools all resolve to plausible paths."""
    tools = {
        name: AndroidTool(name, tmp_path / name)
        for name in ("adb", "emulator", "sdkmanager", "avdmanager")
    }
    tools.update(overrides.pop("tools", {}))
    return AndroidEnvironment(
        host=env.detect_host(system="Darwin", machine="arm64"),
        sdk_root=overrides.pop("sdk_root", tmp_path / "sdk"),
        tools=tools,
        **overrides,
    )


def recorder(responses=None, default=""):
    """A run_tool double that records every command it was given."""
    calls = []

    def fake(executable, arguments, timeout=None, input_text=None):
        calls.append({"executable": str(executable),
                      "arguments": [str(a) for a in arguments],
                      "input": input_text})
        key = " ".join(str(a) for a in arguments)
        for pattern, result in (responses or {}).items():
            if pattern in key:
                return result
        return ToolResult(0, default, "")

    fake.calls = calls
    return fake


def _code_without_prose(path: str) -> str:
    """Module source with comments and every docstring removed."""
    import ast
    import io
    import tokenize

    source = open(path).read()
    kept = []
    previous = tokenize.INDENT
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            continue
        if token.type == tokenize.STRING and previous in (
                tokenize.INDENT, tokenize.NEWLINE, tokenize.NL, tokenize.DEDENT):
            previous = token.type
            continue
        if token.type not in (tokenize.NL, tokenize.NEWLINE):
            previous = token.type
        kept.append(token.string)
    return "\n".join(kept)


# --- pinned configuration ----------------------------------------------------

def test_the_system_image_is_arm64_and_pinned():
    assert emulator.SYSTEM_IMAGE_ABI == "arm64-v8a"
    assert emulator.SYSTEM_IMAGE_PACKAGE == \
        "system-images;android-35;google_apis;arm64-v8a"
    assert emulator.ANDROID_API_LEVEL == 35


def test_the_avd_name_is_a_single_constant():
    assert emulator.AVD_NAME == "DelegateDoctor_ARM64"


def test_the_required_packages_are_explicit():
    assert emulator.EMULATOR_PACKAGES == (
        "platform-tools", "emulator", "platforms;android-35",
        "system-images;android-35;google_apis;arm64-v8a",
    )


def test_the_ndk_is_pinned_not_latest():
    assert emulator.NDK_PACKAGE == "ndk;27.2.12479018"
    assert "latest" not in emulator.NDK_PACKAGE


def test_launch_flags_do_not_throttle_or_destroy():
    flags = emulator.EMULATOR_LAUNCH_FLAGS
    assert "-no-window" in flags and "-no-snapshot" in flags
    # These would either destroy state every run or corrupt the measurement.
    for forbidden in ("-wipe-data", "-netdelay", "-netspeed", "-cores"):
        assert forbidden not in flags


def test_timeouts_are_named_constants_and_bounded():
    assert 0 < emulator.BOOT_TIMEOUT_SECONDS <= 900
    assert 0 < emulator.BOOT_POLL_INTERVAL_SECONDS <= 10
    assert 0 < emulator.ADB_APPEAR_TIMEOUT_SECONDS <= 600


# --- package inspection ------------------------------------------------------

INSTALLED_OUTPUT = """Installed packages:
  Path                 | Version | Description       | Location
  -------              | ------- | -------           | -------
  emulator             | 35.1.4  | Android Emulator  | emulator
  platform-tools       | 35.0.2  | Android SDK Platform-Tools | platform-tools
  platforms;android-35 | 2       | Android SDK Platform 35 | platforms/android-35
"""


def test_installed_packages_are_parsed():
    packages = emulator.parse_installed_packages(INSTALLED_OUTPUT)
    assert "emulator" in packages
    assert "platform-tools" in packages
    assert "platforms;android-35" in packages


def test_available_packages_are_not_counted_as_installed():
    text = INSTALLED_OUTPUT + """
Available Packages:
  Path                                            | Version
  system-images;android-35;google_apis;arm64-v8a  | 1
"""
    packages = emulator.parse_installed_packages(text)
    assert "system-images;android-35;google_apis;arm64-v8a" not in packages


def test_nothing_missing_when_everything_is_installed(tmp_path, monkeypatch):
    complete = INSTALLED_OUTPUT + \
        "  system-images;android-35;google_apis;arm64-v8a | 1 | x | y\n"
    monkeypatch.setattr(emulator, "run_tool",
                        recorder({"--list_installed": ToolResult(0, complete, "")}))
    assert emulator.missing_packages(make_environment(tmp_path)) == []


def test_one_missing_package_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(emulator, "run_tool",
                        recorder({"--list_installed":
                                  ToolResult(0, INSTALLED_OUTPUT, "")}))
    missing = emulator.missing_packages(make_environment(tmp_path))
    assert missing == ["system-images;android-35;google_apis;arm64-v8a"]


def test_several_missing_packages_keep_their_declared_order(tmp_path, monkeypatch):
    monkeypatch.setattr(emulator, "run_tool",
                        recorder({"--list_installed":
                                  ToolResult(0, "Installed packages:\n", "")}))
    missing = emulator.missing_packages(make_environment(tmp_path))
    assert missing == list(emulator.EMULATOR_PACKAGES)


def test_install_names_exactly_the_missing_packages(tmp_path, monkeypatch):
    fake = recorder()
    monkeypatch.setattr(emulator, "run_tool", fake)
    emulator.install_packages(make_environment(tmp_path),
                              ["emulator", "platforms;android-35"],
                              announce=lambda *a: None)
    installed = [call["arguments"] for call in fake.calls]
    assert installed == [["--install", "emulator"],
                         ["--install", "platforms;android-35"]]


def test_installing_nothing_runs_nothing(tmp_path, monkeypatch):
    fake = recorder()
    monkeypatch.setattr(emulator, "run_tool", fake)
    emulator.install_packages(make_environment(tmp_path), [])
    assert fake.calls == []


def test_a_failed_install_mentions_licences(tmp_path, monkeypatch):
    monkeypatch.setattr(emulator, "run_tool",
                        recorder({"--install": ToolResult(1, "", "not accepted")}))
    with pytest.raises(EmulatorError) as caught:
        emulator.install_packages(make_environment(tmp_path), ["emulator"],
                                  announce=lambda *a: None)
    assert "--licenses" in str(caught.value)


# --- licences ----------------------------------------------------------------

def test_licences_are_detected_when_accepted(tmp_path):
    sdk = tmp_path / "sdk"
    (sdk / "licenses").mkdir(parents=True)
    (sdk / "licenses" / "android-sdk-license").write_text("hash")
    assert emulator.licenses_accepted(make_environment(tmp_path, sdk_root=sdk))


def test_missing_licences_are_detected(tmp_path):
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    assert not emulator.licenses_accepted(make_environment(tmp_path, sdk_root=sdk))


def test_delegate_doctor_does_not_accept_licences_on_the_users_behalf():
    """No `yes |` piping: the user accepts Google's terms, not this tool."""
    source = _code_without_prose(emulator.__file__)
    for pattern in ("yes |", '"y\\n" * ', "--licenses\"], input", "accept_licenses("):
        assert pattern not in source, f"emulator.py auto-accepts licences: {pattern}"
    assert "sdkmanager --licenses" in emulator.LICENSES_MESSAGE


# --- AVD ---------------------------------------------------------------------

def test_avds_are_listed_from_the_emulator(tmp_path, monkeypatch):
    fake = recorder({"-list-avds": ToolResult(0, "Pixel_API_34\nDelegateDoctor_ARM64\n", "")})
    monkeypatch.setattr(emulator, "run_tool", fake)
    names = emulator.list_avds(make_environment(tmp_path))
    assert names == ["Pixel_API_34", "DelegateDoctor_ARM64"]
    assert fake.calls[0]["arguments"] == ["-list-avds"]


def test_a_missing_avd_is_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(emulator, "run_tool",
                        recorder({"-list-avds": ToolResult(0, "Pixel_API_34\n", "")}))
    assert not emulator.avd_exists(make_environment(tmp_path))


def test_an_existing_avd_is_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(emulator, "run_tool",
                        recorder({"-list-avds":
                                  ToolResult(0, "DelegateDoctor_ARM64\n", "")}))
    assert emulator.avd_exists(make_environment(tmp_path))


def test_an_unusable_emulator_binary_reports_no_avds(tmp_path, monkeypatch):
    """Listing is a status question; it must not crash a status report."""
    def explode(*args, **kwargs):
        raise EmulatorError("emulator not found")

    monkeypatch.setattr(emulator, "run_tool", explode)
    assert emulator.list_avds(make_environment(tmp_path)) == []


def test_creating_the_avd_names_the_exact_package(tmp_path, monkeypatch):
    fake = recorder({"-list-avds": ToolResult(0, "", "")})
    monkeypatch.setattr(emulator, "run_tool", fake)
    emulator.create_avd(make_environment(tmp_path), announce=lambda *a: None)

    create = [c for c in fake.calls if c["arguments"][:2] == ["create", "avd"]][0]
    assert "--name" in create["arguments"]
    assert emulator.AVD_NAME in create["arguments"]
    assert emulator.SYSTEM_IMAGE_PACKAGE in create["arguments"]
    assert "arm64-v8a" in create["arguments"]


def test_creating_the_avd_answers_the_hardware_prompt(tmp_path, monkeypatch):
    """avdmanager asks about a custom profile; blocking on stdin is not an option."""
    fake = recorder({"-list-avds": ToolResult(0, "", "")})
    monkeypatch.setattr(emulator, "run_tool", fake)
    emulator.create_avd(make_environment(tmp_path), announce=lambda *a: None)
    create = [c for c in fake.calls if c["arguments"][:2] == ["create", "avd"]][0]
    assert create["input"] == "no\n"


def test_an_existing_avd_is_never_recreated(tmp_path, monkeypatch):
    """Idempotence, and the reason --force is not passed."""
    fake = recorder({"-list-avds": ToolResult(0, "DelegateDoctor_ARM64\n", "")})
    monkeypatch.setattr(emulator, "run_tool", fake)
    emulator.create_avd(make_environment(tmp_path), announce=lambda *a: None)
    assert not any(c["arguments"][:2] == ["create", "avd"] for c in fake.calls)


def test_nothing_in_the_module_wipes_or_force_overwrites_an_avd():
    """A user's emulators are not DelegateDoctor's to destroy.

    One deletion exists - `recreate_avd`, for an incompatible AVD that
    DelegateDoctor itself created - and it is guarded by name. Everything else
    on this list would affect state DelegateDoctor does not own, and none of it
    may appear at all.
    """
    code = _code_without_prose(emulator.__file__)
    for destructive in ("--force", "-wipe-data", "shutil.rmtree", '"kill"',
                        "-list-avds -delete", "avd delete --all"):
        assert destructive not in code, f"emulator.py can destroy state: {destructive}"


def test_the_only_deletion_is_guarded_by_delegate_doctors_own_avd_name(tmp_path):
    """`recreate_avd` refuses any AVD DelegateDoctor did not create."""
    environment = make_environment(tmp_path)
    with pytest.raises(emulator.EmulatorError) as caught:
        emulator.recreate_avd(environment, name="Pixel_7_API_34",
                              announce=lambda text: None)
    message = str(caught.value)
    assert "Pixel_7_API_34" in message
    assert "did not create" in message


def test_no_shell_execution_anywhere_in_the_module():
    source = _code_without_prose(emulator.__file__)
    assert "shell=True" not in source
    assert "os.system" not in source


# --- startup -----------------------------------------------------------------

def test_the_emulator_is_launched_with_an_argument_list(tmp_path, monkeypatch):
    recorded = {}

    class FakeProcess:
        pid = 1234

    def fake_popen(command, **kwargs):
        recorded["command"] = command
        recorded["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(emulator.subprocess, "Popen", fake_popen)
    REAL_LAUNCH(make_environment(tmp_path))

    assert isinstance(recorded["command"], list)
    assert "-avd" in recorded["command"]
    assert emulator.AVD_NAME in recorded["command"]
    assert recorded["kwargs"].get("shell") is None      # never a shell


def test_startup_does_not_assume_emulator_5554(tmp_path, monkeypatch):
    """The new serial is discovered by diffing, so an unusual port is fine."""
    before = {"emulator-5554"}          # somebody else's emulator
    serials = [before, before, before | {"emulator-5588"}]

    monkeypatch.setattr(emulator, "_adb_serials", lambda e: serials.pop(0))
    found = emulator.wait_for_new_serial(make_environment(tmp_path), before,
                                         sleep=lambda s: None,
                                         now=iter([0, 1, 2, 3, 4]).__next__)
    assert found == "emulator-5588"


def test_an_unrelated_running_emulator_is_never_adopted(tmp_path, monkeypatch):
    """Nothing new appears, so startup times out rather than stealing a target."""
    monkeypatch.setattr(emulator, "_adb_serials", lambda e: {"emulator-5554"})
    with pytest.raises(EmulatorError) as caught:
        emulator.wait_for_new_serial(make_environment(tmp_path), {"emulator-5554"},
                                     timeout=5, sleep=lambda s: None,
                                     now=iter([0, 1, 2, 10]).__next__)
    assert "did not appear" in str(caught.value)


def test_boot_waits_for_sys_boot_completed(tmp_path, monkeypatch):
    """`adb wait-for-device` is not enough; the property is what matters."""
    answers = iter(["", "", "1"])
    asked = []

    def fake_property(environment, serial, name):
        asked.append(name)
        return next(answers)

    monkeypatch.setattr(emulator, "_adb_property", fake_property)
    emulator.wait_for_boot(make_environment(tmp_path), "emulator-5556",
                           sleep=lambda s: None,
                           now=iter([0, 1, 2, 3, 4, 5]).__next__)
    assert asked == ["sys.boot_completed"] * 3


def test_a_boot_that_never_completes_times_out(tmp_path, monkeypatch):
    monkeypatch.setattr(emulator, "_adb_property", lambda e, s, n: "")
    with pytest.raises(EmulatorError) as caught:
        emulator.wait_for_boot(make_environment(tmp_path), "emulator-5556",
                               timeout=10, sleep=lambda s: None,
                               now=iter([0, 5, 20]).__next__)
    assert "did not finish booting" in str(caught.value)


def test_an_arm64_emulator_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(emulator, "_adb_property", lambda e, s, n: "arm64-v8a")
    assert emulator.verify_abi(make_environment(tmp_path), "emulator-5556") == \
        "arm64-v8a"


def test_an_x86_emulator_is_rejected_after_boot(tmp_path, monkeypatch):
    """The last line of defence: a booted x86 image is still not an Arm target."""
    monkeypatch.setattr(emulator, "_adb_property", lambda e, s, n: "x86_64")
    with pytest.raises(EmulatorError) as caught:
        emulator.verify_abi(make_environment(tmp_path), "emulator-5556")
    message = str(caught.value)
    assert "ARM64 EMULATOR UNAVAILABLE" in message
    assert "never used as a benchmark target" in message


def test_starting_a_missing_avd_says_how_to_create_it(tmp_path, monkeypatch):
    monkeypatch.setattr(emulator, "list_avds", lambda e: [])
    with pytest.raises(EmulatorError) as caught:
        emulator.start_delegate_doctor_emulator(make_environment(tmp_path),
                                                announce=lambda *a: None)
    assert "delegate-doctor setup-android" in str(caught.value)


def test_a_full_start_returns_the_discovered_serial(tmp_path, monkeypatch):
    monkeypatch.setattr(emulator, "list_avds", lambda e: [emulator.AVD_NAME])
    monkeypatch.setattr(emulator, "_adb_serials", lambda e: set())
    monkeypatch.setattr(emulator, "launch_emulator_process", lambda e, n=None: None)
    monkeypatch.setattr(emulator, "wait_for_new_serial",
                        lambda *a, **k: "emulator-5560")
    monkeypatch.setattr(emulator, "wait_for_boot", lambda *a, **k: None)
    monkeypatch.setattr(emulator, "verify_abi", lambda e, s: "arm64-v8a")

    serial = emulator.start_delegate_doctor_emulator(make_environment(tmp_path),
                                                     announce=lambda *a: None)
    assert serial == "emulator-5560"
