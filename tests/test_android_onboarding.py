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

from delegate_doctor import (android_environment, android_packages, android_setup,
                             cli, device, environment_check, target_selection)

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




def test_windows_resolves_adb_exe(tmp_path, monkeypatch):
    sdk = make_sdk(tmp_path / "sdk", windows=True)
    use_sdk(monkeypatch, sdk, path_adb=None, windows=True)

    resolved = device.resolve_adb()
    assert resolved.endswith("adb.exe")
    assert "platform-tools" in resolved




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






# --- package split ------------------------------------------------------------

SYSTEM_IMAGE = "system-images;android-35;google_apis;arm64-v8a"





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















def test_check_passes_for_adb_without_it_being_on_path(tmp_path, monkeypatch):
    """`which adb` returning nothing must not fail a ready machine."""
    sdk = make_sdk(tmp_path / "sdk")
    monkeypatch.setenv("ANDROID_HOME", str(sdk))
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    monkeypatch.setattr(device.shutil, "which", lambda name: None)

    assert environment_check.check_adb(path_lookup=lambda name: None).status \
        == environment_check.PASS




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





# --- the emulator is gone, and must stay gone --------------------------------

def test_setup_provisions_only_adb_and_the_ndk():
    """No emulator package, no SDK platform, no system image."""
    assert android_packages.REQUIRED_PACKAGES == (
        "platform-tools", android_packages.NDK_PACKAGE)
    for gone in ("emulator", "platforms;android-35",
                 "system-images;android-35;google_apis;arm64-v8a"):
        assert gone not in android_packages.REQUIRED_PACKAGES


def test_no_production_module_can_create_or_boot_an_avd():
    """The managed-AVD feature is removed, not merely unused."""
    import pathlib

    root = pathlib.Path(android_setup.__file__).parent
    for path in root.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        source = path.read_text()
        for gone in ("DelegateDoctor_ARM64", "avdmanager create",
                     "launch_emulator_process", "start_delegate_doctor_emulator",
                     "system-images;android-35"):
            assert gone not in source, f"{path.name} still references {gone}"


def test_the_emulator_module_no_longer_exists():
    with pytest.raises(ModuleNotFoundError):
        __import__("delegate_doctor.emulator")


def test_setup_android_takes_no_emulator_flag():
    with pytest.raises(SystemExit):
        cli.main(["setup-android", "--emulator"])


def test_setup_help_says_nothing_is_downloaded_for_an_emulator(capsys):
    with pytest.raises(SystemExit):
        cli.main(["setup-android", "--help"])
    text = capsys.readouterr().out
    assert "--emulator" not in text
    assert "no AVD" in text or "no emulator" in text


# --- benchmark defaults ------------------------------------------------------

def test_the_benchmark_defaults_are_five_twenty_one():
    """One canonical set, shared by the CLI, the API and the benchmark itself.

    A CLI default that disagreed with the pipeline default would mean the
    published numbers came from settings nobody documented.
    """
    import inspect

    from delegate_doctor import benchmarking, pipeline

    for function in (pipeline.run_optimization, benchmarking.benchmark_before_after):
        defaults = inspect.signature(function).parameters
        assert defaults["warmup_iterations"].default == 5
        assert defaults["measured_iterations"].default == 20
        assert defaults["repetitions"].default == 1


def test_the_cli_defaults_match_the_api(capsys):
    with pytest.raises(SystemExit):
        cli.main(["optimize", "--help"])
    text = capsys.readouterr().out
    assert "--warmup" in text and "--iters" in text and "--reps" in text


# --- examples ----------------------------------------------------------------

def test_no_example_downloads_pretrained_weights():
    """Every checked-in example must run offline.

    A first run that reaches the PyTorch hub is a first run that fails behind a
    firewall, and none of these examples needs trained weights: delegation is
    decided by the graph.
    """
    import pathlib
    import re

    root = pathlib.Path(android_setup.__file__).parent.parent / "examples"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        source = path.read_text()
        # Strip docstrings/comments: several examples legitimately *explain*
        # that they avoid pretrained weights.
        code = "\n".join(line for line in source.splitlines()
                         if not line.strip().startswith("#"))
        for pattern in (r"weights\s*=\s*[A-Za-z_]+_Weights",
                        r"weights\s*=\s*[\"']DEFAULT[\"']",
                        r"pretrained\s*=\s*True",
                        r"encoder_weights\s*=\s*[\"'][A-Za-z]",
                        r"load_state_dict_from_url",
                        r"torch\.hub"):
            if re.search(pattern, code):
                offenders.append(f"{path.name}: {pattern}")
    assert offenders == [], offenders


# --- documentation stays in step ---------------------------------------------

def test_the_readme_has_no_stale_target_or_emulator_commands():
    import pathlib

    root = pathlib.Path(android_setup.__file__).parent.parent
    text = (root / "README.md").read_text()
    for gone in ("--target device", "--target emulator", "--target auto",
                 "setup-android --emulator", "DelegateDoctor_ARM64",
                 "system-images;android-35"):
        assert gone not in text, f"README still shows {gone}"
