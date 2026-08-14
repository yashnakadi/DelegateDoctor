"""Physical-device-first Android onboarding, and adb that does not need PATH.

Two behaviours are pinned here.

**adb is resolved, not assumed.** Android Studio installs adb at
`<sdk>/platform-tools/adb` and does not add it to PATH. DelegateDoctor used to
invoke a bare `adb`, so a user with a working SDK and a connected phone was
told "No Arm64 Android target is attached" - a device problem reported for a
PATH problem.

**The emulator system image is opt-in.** It is a multi-gigabyte download, and
plain `setup-android` must never fetch it: the fast path is a phone on USB.

Fully offline: no sdkmanager, no adb, no emulator, no device, no network.
"""

import pytest

from delegate_doctor import (android_environment, android_setup, cli, device,
                             emulator, environment_check, target_selection)

# Captured at import, before conftest's autouse guard replaces it. These tests
# exercise adb resolution itself, so they need the real implementation back -
# with subprocess mocked, so still nothing runs.
REAL_RUN_ADB = device.run_adb


# --- fake SDKs and fake adb output -------------------------------------------

def make_sdk(root, windows=False):
    """A directory tree that looks like an Android Studio SDK."""
    suffix = ".exe" if windows else ""
    (root / "platform-tools").mkdir(parents=True)
    (root / "platform-tools" / f"adb{suffix}").write_text("x")
    return root


@pytest.fixture(autouse=True)
def clear_adb_cache():
    """adb is resolved once per process; each test starts from nothing."""
    device.reset_adb_cache()
    yield
    device.reset_adb_cache()


def use_sdk(monkeypatch, sdk_root, path_adb=None, windows=False):
    """Point SDK discovery at `sdk_root`, and PATH at `path_adb` or nothing."""
    host = android_environment.detect_host(
        system="Windows" if windows else "Darwin",
        machine="arm64")
    monkeypatch.setattr(android_environment, "detect_host", lambda **kw: host)
    monkeypatch.setenv("ANDROID_HOME", str(sdk_root))
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    monkeypatch.setattr(device.shutil, "which", lambda name: path_adb)


class FakeAdb:
    """Records every adb invocation and replays scripted output.

    Substitutes for `subprocess.run`, so the argv DelegateDoctor builds - and
    in particular which executable it names - is what gets asserted.
    """

    def __init__(self, devices_output="List of devices attached\n", properties=None):
        self.calls = []
        self.devices_output = devices_output
        self.properties = properties or {}

    def __call__(self, command, capture_output=True, text=True, check=True):
        self.calls.append(list(command))
        arguments = command[1:]
        if arguments and arguments[0] == "-s":
            arguments = arguments[2:]
        if arguments[:1] == ["devices"]:
            return _Completed(self.devices_output)
        if arguments[:2] == ["shell", "getprop"]:
            return _Completed(self.properties.get(arguments[2], "") + "\n")
        # `run_on_device` appends `; echo DD_EXIT=$?` and checks for it.
        return _Completed("DD_EXIT=0\n")

    @property
    def executables(self):
        return {call[0] for call in self.calls}


class _Completed:
    def __init__(self, stdout=""):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = 0


def phone_properties(abi="arm64-v8a", model="RMX2030"):
    return {"ro.product.model": model, "ro.product.cpu.abi": abi,
            "ro.build.version.release": "10", "ro.build.version.sdk": "29",
            "ro.hardware": "qcom", "ro.product.manufacturer": "realme",
            "ro.boot.qemu.avd_name": "", "ro.kernel.qemu.avd_name": ""}


# --- adb resolution ----------------------------------------------------------

def test_sdk_adb_is_found_when_path_has_none(tmp_path, monkeypatch):
    """The reported failure, as a regression test."""
    sdk = make_sdk(tmp_path / "sdk")
    use_sdk(monkeypatch, sdk, path_adb=None)

    assert device.resolve_adb() == str(sdk / "platform-tools" / "adb")


def test_sdk_adb_is_preferred_over_a_different_adb_on_path(tmp_path, monkeypatch):
    """Provisioning one SDK while talking to another produces impossible bugs."""
    sdk = make_sdk(tmp_path / "sdk")
    other = tmp_path / "somewhere-else" / "adb"
    other.parent.mkdir(parents=True)
    other.write_text("x")
    use_sdk(monkeypatch, sdk, path_adb=str(other))

    assert device.resolve_adb() == str(sdk / "platform-tools" / "adb")


def test_path_adb_is_the_fallback_when_no_sdk_has_one(tmp_path, monkeypatch):
    empty = tmp_path / "sdk"
    (empty / "emulator").mkdir(parents=True)
    fallback = tmp_path / "bin" / "adb"
    fallback.parent.mkdir(parents=True)
    fallback.write_text("x")
    use_sdk(monkeypatch, empty, path_adb=str(fallback))

    assert device.resolve_adb() == str(fallback)


def test_windows_resolves_adb_exe(tmp_path, monkeypatch):
    sdk = make_sdk(tmp_path / "sdk", windows=True)
    use_sdk(monkeypatch, sdk, path_adb=None, windows=True)

    resolved = device.resolve_adb()
    assert resolved.endswith("adb.exe")
    assert "platform-tools" in resolved


def test_no_adb_anywhere_is_reported_as_adb_missing(tmp_path, monkeypatch):
    """Not as "device unavailable" - a different problem with a different fix."""
    empty = tmp_path / "sdk"
    (empty / "emulator").mkdir(parents=True)
    use_sdk(monkeypatch, empty, path_adb=None)

    assert device.resolve_adb() is None
    with pytest.raises(device.AdbNotFound) as raised:
        device.require_adb()
    assert "not a device problem" in str(raised.value)
    # Still a DeviceError, so every existing handler keeps working.
    assert isinstance(raised.value, device.DeviceError)


def test_every_adb_call_uses_the_resolved_executable(tmp_path, monkeypatch):
    sdk = make_sdk(tmp_path / "sdk")
    use_sdk(monkeypatch, sdk, path_adb=None)
    fake = FakeAdb()
    monkeypatch.setattr(device.subprocess, "run", fake)
    monkeypatch.setattr(device, "run_adb", REAL_RUN_ADB)

    REAL_RUN_ADB("devices")
    device.run_on_device("true", serial="X")

    expected = str(sdk / "platform-tools" / "adb")
    assert fake.executables == {expected}, "something still invoked a bare adb"


# --- device classification ----------------------------------------------------

def attach(monkeypatch, tmp_path, listing, properties=None):
    sdk = make_sdk(tmp_path / "sdk")
    use_sdk(monkeypatch, sdk, path_adb=None)
    fake = FakeAdb(devices_output=listing, properties=properties)
    monkeypatch.setattr(device.subprocess, "run", fake)
    # Put the real run_adb back over conftest's refusal, for these tests only.
    monkeypatch.setattr(device, "run_adb", REAL_RUN_ADB)
    monkeypatch.setattr(target_selection, "run_adb", REAL_RUN_ADB)
    return fake


def test_a_physical_arm64_phone_is_discovered_through_the_sdk_adb(tmp_path,
                                                                  monkeypatch):
    attach(monkeypatch, tmp_path,
           "List of devices attached\nR9AB123\tdevice\n",
           phone_properties())

    status = target_selection.physical_device_status()
    assert status.ready
    assert status.target.info.abi == "arm64-v8a"


def test_an_unauthorized_device_is_its_own_outcome(tmp_path, monkeypatch):
    attach(monkeypatch, tmp_path,
           "List of devices attached\nR9AB123\tunauthorized\n")

    status = target_selection.physical_device_status()
    assert status.status == target_selection.PHYSICAL_UNAUTHORIZED
    assert "R9AB123" in status.detail
    assert "USB debugging" in android_setup.format_physical_device_summary(status)


def test_an_offline_device_is_its_own_outcome(tmp_path, monkeypatch):
    attach(monkeypatch, tmp_path,
           "List of devices attached\nR9AB123\toffline\n")

    status = target_selection.physical_device_status()
    assert status.status == target_selection.PHYSICAL_OFFLINE
    assert "OFFLINE" in android_setup.format_physical_device_summary(status)


def test_an_x86_device_is_not_arm_evidence(tmp_path, monkeypatch):
    attach(monkeypatch, tmp_path,
           "List of devices attached\nR9AB123\tdevice\n",
           phone_properties(abi="x86_64"))

    status = target_selection.physical_device_status()
    assert status.status == target_selection.PHYSICAL_WRONG_ABI
    summary = android_setup.format_physical_device_summary(status)
    assert "x86_64" in summary and "arm64-v8a" in summary


def test_no_device_is_not_a_setup_failure(tmp_path, monkeypatch):
    attach(monkeypatch, tmp_path, "List of devices attached\n")

    status = target_selection.physical_device_status()
    assert status.status == target_selection.PHYSICAL_NONE
    summary = android_setup.format_physical_device_summary(status)
    assert "Common Android environment ready" in summary
    assert "setup-android --emulator" in summary
    assert "large" in summary.lower()


def test_missing_adb_summary_does_not_blame_the_device(tmp_path, monkeypatch):
    empty = tmp_path / "sdk"
    (empty / "emulator").mkdir(parents=True)
    use_sdk(monkeypatch, empty, path_adb=None)
    monkeypatch.setattr(target_selection, "run_adb", REAL_RUN_ADB)

    status = target_selection.physical_device_status()
    assert status.status == target_selection.PHYSICAL_NO_ADB
    summary = android_setup.format_physical_device_summary(status)
    assert "ADB" in summary and "NOT FOUND" in summary
    assert "not a device problem" in summary or "cannot\nlook for a device" in summary


# --- package split ------------------------------------------------------------

SYSTEM_IMAGE = "system-images;android-35;google_apis;arm64-v8a"


def test_the_system_image_is_not_a_common_package():
    assert SYSTEM_IMAGE not in emulator.COMMON_PACKAGES
    assert SYSTEM_IMAGE in emulator.EMULATOR_PACKAGES


def test_common_packages_are_what_a_phone_and_the_runner_build_need():
    assert "platform-tools" in emulator.COMMON_PACKAGES      # adb
    assert emulator.NDK_PACKAGE in emulator.COMMON_PACKAGES  # cross-compile
    assert "emulator" not in emulator.COMMON_PACKAGES


def test_the_emulator_package_set_is_everything_else():
    assert set(emulator.EMULATOR_PACKAGES) == {
        "emulator", "platforms;android-35", SYSTEM_IMAGE}


class Provisioning:
    """Records which packages were asked for and installed."""

    def __init__(self, installed=()):
        self.installed = list(installed)
        self.requested = []
        self.performed = []

    def missing(self, environment, required=()):
        self.requested.append(list(required))
        return [p for p in required if p not in self.installed]

    def install(self, environment, packages, announce=print):
        self.performed += list(packages)
        self.installed += list(packages)


def provisioning(monkeypatch, installed=()):
    fake = Provisioning(installed)
    monkeypatch.setattr(emulator, "missing_packages", fake.missing)
    monkeypatch.setattr(emulator, "install_packages", fake.install)
    monkeypatch.setattr(android_setup, "ensure_licenses",
                        lambda *a, **k: True)
    return fake


def environment_for(tmp_path):
    sdk = make_sdk(tmp_path / "sdk")
    tools = sdk / "cmdline-tools" / "latest" / "bin"
    tools.mkdir(parents=True)
    for name in ("sdkmanager", "avdmanager"):
        (tools / name).write_text("x")
    (sdk / "emulator").mkdir()
    (sdk / "emulator" / "emulator").write_text("x")
    return android_environment.detect(
        environment={"ANDROID_HOME": str(sdk)},
        host=android_environment.detect_host(system="Darwin", machine="arm64"),
        path_lookup=lambda name: None)


def test_common_provisioning_never_requests_the_system_image(tmp_path,
                                                             monkeypatch):
    fake = provisioning(monkeypatch)
    android_setup.provision_common(environment_for(tmp_path), interactive=False,
                                   assume_yes=True, announce=lambda *a: None)

    for request in fake.requested:
        assert SYSTEM_IMAGE not in request
    assert SYSTEM_IMAGE not in fake.performed


def test_emulator_provisioning_does_request_the_system_image(tmp_path,
                                                             monkeypatch):
    fake = provisioning(monkeypatch)
    monkeypatch.setattr(emulator, "avd_exists", lambda environment: True)
    monkeypatch.setattr(emulator, "avd_is_compatible",
                        lambda environment: (True, ""))

    android_setup.provision_emulator(environment_for(tmp_path),
                                     interactive=False, assume_yes=True,
                                     announce=lambda *a: None)
    assert SYSTEM_IMAGE in fake.performed


def test_host_support_is_checked_before_the_large_download(tmp_path, monkeypatch):
    """An unsupported host learns so in a second, not after a long download."""
    fake = provisioning(monkeypatch)
    intel = android_environment.detect(
        environment={"ANDROID_HOME": str(make_sdk(tmp_path / "sdk"))},
        host=android_environment.detect_host(system="Darwin", machine="x86_64"),
        path_lookup=lambda name: None)

    assert not android_setup.provision_emulator(
        intel, interactive=False, assume_yes=True, announce=lambda *a: None)
    assert fake.requested == [], "packages were inspected on an unsupported host"
    assert fake.performed == []


def test_an_existing_avd_is_reused_rather_than_recreated(tmp_path, monkeypatch):
    provisioning(monkeypatch, installed=list(emulator.EMULATOR_PACKAGES))
    monkeypatch.setattr(emulator, "avd_exists", lambda environment: True)
    monkeypatch.setattr(emulator, "avd_is_compatible",
                        lambda environment: (True, ""))
    recreated = []
    monkeypatch.setattr(emulator, "recreate_avd",
                        lambda *a, **k: recreated.append(1))
    monkeypatch.setattr(emulator, "create_avd",
                        lambda *a, **k: recreated.append(1))

    assert android_setup.provision_emulator(
        environment_for(tmp_path), interactive=False, assume_yes=True,
        announce=lambda *a: None)
    assert recreated == [], "an idempotent re-run touched the AVD"


def test_only_the_managed_avd_is_ever_named():
    """Unrelated AVDs must not be a thing DelegateDoctor knows how to touch."""
    import inspect

    source = inspect.getsource(android_setup.provision_emulator)
    assert "AVD_NAME" in source
    assert emulator.AVD_NAME == "DelegateDoctor_ARM64"


def test_an_emulator_failure_does_not_erase_common_setup():
    message = android_setup.format_emulator_failure("sdkmanager exploded")
    assert "Common Android environment is ready" in message
    assert "sdkmanager exploded" in message, "the real reason was hidden"
    assert "setup-android --emulator" in message
    assert "--target device" in message


def test_the_large_download_is_announced_before_it_starts(tmp_path, monkeypatch):
    said = []
    monkeypatch.setattr(emulator, "run_tool",
                        lambda *a, **k: type("R", (), {"ok": True, "stdout": "",
                                                       "stderr": ""})())
    environment = environment_for(tmp_path)
    monkeypatch.setattr(type(environment), "tool_path",
                        lambda self, name: tmp_path / name)

    emulator.install_packages(environment, [SYSTEM_IMAGE], announce=said.append)
    text = "\n".join(said)
    assert "large download" in text
    assert SYSTEM_IMAGE in text


# --- CLI ----------------------------------------------------------------------

def test_the_parser_accepts_the_emulator_flag(monkeypatch):
    seen = {}
    monkeypatch.setattr(android_setup, "setup_android_runners",
                        lambda **kwargs: seen.update(kwargs) or 0)

    assert cli.main(["setup-android", "--emulator"]) == 0
    assert seen["setup_emulator"] is True


def test_plain_setup_does_not_request_the_emulator(monkeypatch):
    seen = {}
    monkeypatch.setattr(android_setup, "setup_android_runners",
                        lambda **kwargs: seen.update(kwargs) or 0)

    assert cli.main(["setup-android"]) == 0
    assert seen["setup_emulator"] is False


def test_the_deprecated_skip_flag_still_parses(monkeypatch):
    """Kept so an existing script does not start failing on an unknown flag."""
    seen = {}
    monkeypatch.setattr(android_setup, "setup_android_runners",
                        lambda **kwargs: seen.update(kwargs) or 0)

    assert cli.main(["setup-android", "--skip-emulator"]) == 0
    assert seen["skip_emulator"] is True


def test_setup_help_distinguishes_the_two_paths(capsys):
    with pytest.raises(SystemExit):
        cli.main(["setup-android", "--help"])
    text = capsys.readouterr().out

    assert "--emulator" in text
    assert "LARGE" in text or "large" in text
    assert "does NOT download" in text or "not download" in text.lower()


def test_no_help_text_claims_the_emulator_is_automatic(capsys):
    with pytest.raises(SystemExit):
        cli.main(["setup-android", "--help"])
    text = capsys.readouterr().out.lower()
    for stale in ("emulator, where the host supports it\n  the two cross",
                  "installed privately"):
        assert stale not in text


# --- environment check --------------------------------------------------------

def test_check_passes_for_adb_without_it_being_on_path(tmp_path, monkeypatch):
    """`which adb` returning nothing must not fail a ready machine."""
    sdk = make_sdk(tmp_path / "sdk")
    monkeypatch.setenv("ANDROID_HOME", str(sdk))
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    monkeypatch.setattr(device.shutil, "which", lambda name: None)

    assert environment_check.check_adb(path_lookup=lambda name: None).status \
        == environment_check.PASS


def test_check_reports_adb_missing_when_there_is_none(tmp_path, monkeypatch):
    empty = tmp_path / "sdk"
    (empty / "emulator").mkdir(parents=True)
    monkeypatch.setenv("ANDROID_HOME", str(empty))
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    monkeypatch.setattr(device.shutil, "which", lambda name: None)

    result = environment_check.check_adb(path_lookup=lambda name: None)
    assert result.status == environment_check.MISSING
    assert "setup-android" in result.remedy


def test_the_normal_check_line_is_concise(tmp_path, monkeypatch):
    """The path belongs in --verbose, not in every run's output."""
    sdk = make_sdk(tmp_path / "sdk")
    monkeypatch.setenv("ANDROID_HOME", str(sdk))
    monkeypatch.setattr(device.shutil, "which", lambda name: None)

    result = environment_check.check_adb(path_lookup=lambda name: None)
    assert result.detail == ""


def test_a_displayed_path_hides_the_home_directory():
    from pathlib import Path

    shown = environment_check.describe_path(Path.home() / "Library" / "adb")
    assert str(Path.home()) not in shown
    assert shown.startswith("~")


# --- the closing block --------------------------------------------------------

def test_plain_setup_does_not_report_a_missing_avd_as_a_failure(monkeypatch):
    """A fresh machine has no AVD, because plain setup deliberately made none.

    Printing "Managed Arm64 environment UNAVAILABLE - the AVD does not exist"
    would report the deliberate absence of an unrequested component as a
    failure, right after telling the user setup succeeded.
    """
    monkeypatch.setattr(android_setup, "managed_environment_ready",
                        lambda runners, environment: False)
    monkeypatch.setattr(target_selection, "physical_device_status",
                        lambda: target_selection.PhysicalDeviceStatus(
                            target_selection.PHYSICAL_NONE))
    said = []
    android_setup._report_closing(False, "runners", said.append)

    text = "\n".join(said)
    assert "Common Android environment ready" in text
    assert "UNAVAILABLE" not in text
    assert "AVD does not exist" not in text


def test_plain_setup_still_mentions_an_emulator_that_is_ready(monkeypatch):
    """If the AVD happens to exist, saying so is useful rather than misleading."""
    monkeypatch.setattr(android_setup, "managed_environment_ready",
                        lambda runners, environment: True)
    monkeypatch.setattr(target_selection, "physical_device_status",
                        lambda: target_selection.PhysicalDeviceStatus(
                            target_selection.PHYSICAL_NONE))
    said = []
    android_setup._report_closing(False, "runners", said.append)

    assert "--target emulator" in "\n".join(said)


def test_the_emulator_path_closes_with_the_managed_verdict(monkeypatch):
    monkeypatch.setattr(android_setup, "format_verdict",
                        lambda runners: "VERDICT")
    checked = []
    monkeypatch.setattr(target_selection, "physical_device_status",
                        lambda: checked.append(1))
    said = []
    android_setup._report_closing(True, "runners", said.append)

    assert said == ["VERDICT"]
    assert checked == [], "--emulator should not answer a phone question"
