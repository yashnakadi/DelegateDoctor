"""One-command Android setup, on top of the SDK Android Studio installed.

Installing Android Studio and completing its Setup Wizard is the one manual
Android step. Everything after it - the pinned packages, the licences, the
AVD, the runners - is DelegateDoctor's job.

DelegateDoctor used to be able to bootstrap its own SDK from Google's
command-line-tools archive. That is gone: two ways to obtain an SDK meant two
onboarding paths, two sets of failure modes, and a checksum table that could
not be verified offline. A machine with no SDK is now told to install Android
Studio, and setup stops there.

Nothing here touches the network, the filesystem outside `tmp_path`, or a real
SDK. The negatives matter as much as the positives: an x86_64 host is never
told it has an Arm benchmark, no AVD but DelegateDoctor's own is ever touched,
and no download is ever attempted.
"""

import io
import zipfile
from pathlib import Path

import pytest

from delegate_doctor import android_environment as env
from delegate_doctor import android_setup, emulator
from delegate_doctor.android_environment import AndroidEnvironment, AndroidTool

# --- where things live --------------------------------------------------------








# --- discovery order --------------------------------------------------------------


def test_android_home_wins_over_everything(tmp_path, monkeypatch):
    """Case 12: the explicit variable is honoured over any standard location."""
    explicit = tmp_path / "explicit"
    (explicit / "platform-tools").mkdir(parents=True)
    standard = tmp_path / "standard"
    (standard / "platform-tools").mkdir(parents=True)
    monkeypatch.setattr(env, "DEFAULT_SDK_LOCATIONS", (str(standard),))

    assert env.find_sdk_root({"ANDROID_HOME": str(explicit)},
                             home=tmp_path) == explicit


def test_android_sdk_root_is_consulted_after_android_home(tmp_path):
    fallback = tmp_path / "fallback"
    (fallback / "platform-tools").mkdir(parents=True)
    found = env.find_sdk_root({"ANDROID_SDK_ROOT": str(fallback)}, home=tmp_path)
    assert found == fallback






def test_an_empty_directory_does_not_count_as_an_sdk(tmp_path):
    """A folder surviving an uninstall must not read as a usable SDK."""
    (tmp_path / "empty").mkdir()
    assert not env.is_usable_sdk(tmp_path / "empty")
    (tmp_path / "empty" / "platform-tools").mkdir()
    assert env.is_usable_sdk(tmp_path / "empty")


# --- reusing what is already there --------------------------------------------------


def make_environment(tmp_path, sdk_root=None, tools=("adb", "emulator",
                                                     "sdkmanager", "avdmanager"),
                     system="Darwin", machine="arm64"):
    return AndroidEnvironment(
        host=env.detect_host(system=system, machine=machine),
        sdk_root=sdk_root,
        tools={name: AndroidTool(name, tmp_path / name) for name in tools},
    )




def test_a_partial_sdk_installs_only_what_is_missing(tmp_path, monkeypatch):
    """Existing packages are left alone; only the gaps are filled."""
    installed = []
    monkeypatch.setattr(emulator, "installed_packages",
                        lambda environment: {"platform-tools", "emulator"})
    monkeypatch.setattr(emulator, "install_packages",
                        lambda environment, packages, announce=print:
                        installed.extend(packages))
    monkeypatch.setattr(emulator, "licenses_accepted", lambda environment: True)
    monkeypatch.setattr(emulator, "avd_exists", lambda environment, name=None: True)
    monkeypatch.setattr(emulator, "avd_is_compatible",
                        lambda environment, name=None, home=None: (True, ""))

    environment = make_environment(tmp_path, sdk_root=tmp_path / "sdk")
    android_setup.provision_emulator(environment, interactive=True,
                                     assume_yes=True,
                                     announce=lambda text: None)
    assert "platform-tools" not in installed
    assert "emulator" not in installed
    assert any("system-images" in package for package in installed)






# --- licences ------------------------------------------------------------------------


def test_unaccepted_licences_stop_a_non_interactive_run(tmp_path, monkeypatch):
    """No flag synthesizes agreement, and none is offered."""
    monkeypatch.setattr(emulator, "licenses_accepted", lambda environment: False)
    said = []
    accepted = android_setup.ensure_licenses(
        make_environment(tmp_path, sdk_root=tmp_path / "sdk"),
        interactive=False, announce=said.append)
    assert accepted is False
    text = "\n".join(said)
    assert "cannot ask" in text
    assert "sdkmanager --licenses" in text


def test_no_flag_can_auto_accept_licences():
    """`--yes` installs packages. It has never meant "I agree to the licences"."""
    import inspect

    source = inspect.getsource(emulator)
    # Piping agreement into the licence prompt is the specific thing forbidden.
    assert 'input_text="y' not in source
    assert '"yes\\n" * ' not in source


def test_the_licence_process_gets_the_users_terminal(tmp_path, monkeypatch):
    """Not captured: a user cannot agree to text they were never shown."""
    monkeypatch.setattr(emulator, "licenses_accepted", lambda environment: True)
    commands = []
    emulator.accept_licenses_interactively(
        make_environment(tmp_path, sdk_root=tmp_path / "sdk"),
        announce=lambda text: None,
        prompt=lambda question: "",
        runner=lambda command: commands.append(command) or 0)
    assert commands and commands[0][-1] == "--licenses"


# --- AVD safety -----------------------------------------------------------------------


def test_only_delegate_doctors_own_avd_may_be_recreated(tmp_path):
    with pytest.raises(emulator.EmulatorError) as caught:
        emulator.recreate_avd(make_environment(tmp_path),
                              name="My_Pixel_AVD", announce=lambda text: None)
    assert "did not create" in str(caught.value)


def test_a_compatible_managed_avd_is_reused(tmp_path, monkeypatch):
    config = (tmp_path / ".android" / "avd" /
              f"{emulator.AVD_NAME}.avd" / "config.ini")
    config.parent.mkdir(parents=True)
    config.write_text("abi.type=arm64-v8a\nhw.cpu.arch=arm64\n")

    compatible, reason = emulator.avd_is_compatible(
        make_environment(tmp_path), home=tmp_path)
    assert compatible
    assert reason == ""


def test_an_x86_managed_avd_is_reported_as_incompatible(tmp_path):
    config = (tmp_path / ".android" / "avd" /
              f"{emulator.AVD_NAME}.avd" / "config.ini")
    config.parent.mkdir(parents=True)
    config.write_text("abi.type=x86_64\n")

    compatible, reason = emulator.avd_is_compatible(
        make_environment(tmp_path), home=tmp_path)
    assert not compatible
    assert "not Arm evidence" in reason


def test_an_unreadable_avd_config_is_not_called_incompatible(tmp_path):
    """Absence of evidence is not evidence of a mismatch."""
    compatible, _ = emulator.avd_is_compatible(make_environment(tmp_path),
                                               home=tmp_path)
    assert compatible


# --- host policy --------------------------------------------------------------------


@pytest.mark.parametrize("system, machine, expected", [
    ("Darwin", "arm64", env.SUPPORT_VALIDATED),
    ("Linux", "aarch64", env.SUPPORT_UNTESTED),
    ("Windows", "arm64", env.SUPPORT_UNTESTED),
    ("Darwin", "x86_64", env.SUPPORT_UNAVAILABLE),
    ("Linux", "x86_64", env.SUPPORT_UNAVAILABLE),
    ("Windows", "AMD64", env.SUPPORT_UNAVAILABLE),
])
def test_arm_emulator_support_is_three_valued(system, machine, expected):
    host = env.detect_host(system=system, machine=machine)
    assert env.emulator_support(host)[0] == expected


def test_an_x86_host_is_never_reported_ready_even_with_a_working_emulator(
        tmp_path, monkeypatch):
    """An x86_64 emulator is not Arm evidence, however well it runs."""
    monkeypatch.setattr(emulator, "avd_exists", lambda environment, name=None: True)
    runners = tmp_path / "runners"
    runners.mkdir()
    monkeypatch.setattr(android_setup, "runners_already_installed",
                        lambda directory: True)

    environment = make_environment(tmp_path, sdk_root=tmp_path / "sdk",
                                   system="Linux", machine="x86_64")
    assert not android_setup.managed_environment_ready(runners, environment)

    verdict = android_setup.format_verdict(runners, environment)
    assert android_setup.UNAVAILABLE in verdict
    assert "physical arm64-v8a Android device" in verdict


def test_an_arm_host_with_everything_present_is_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(emulator, "avd_exists", lambda environment, name=None: True)
    monkeypatch.setattr(android_setup, "runners_already_installed",
                        lambda directory: True)

    environment = make_environment(tmp_path, sdk_root=tmp_path / "sdk")
    assert android_setup.managed_environment_ready(tmp_path / "runners",
                                                   environment)
    verdict = android_setup.format_verdict(tmp_path / "runners", environment)
    assert android_setup.READY in verdict
    assert "--target emulator" in verdict


def test_missing_runners_are_never_reported_ready(tmp_path, monkeypatch):
    """READY must mean a run can start now, not "almost"."""
    monkeypatch.setattr(emulator, "avd_exists", lambda environment, name=None: True)
    monkeypatch.setattr(android_setup, "runners_already_installed",
                        lambda directory: False)

    environment = make_environment(tmp_path, sdk_root=tmp_path / "sdk")
    assert not android_setup.managed_environment_ready(tmp_path / "runners",
                                                       environment)
    assert "runners are not built" in android_setup.format_verdict(
        tmp_path / "runners", environment)


# --- optimize can trigger setup --------------------------------------------------------


def test_optimize_offers_setup_when_the_emulator_is_missing(monkeypatch):
    from delegate_doctor import cli

    monkeypatch.setattr(android_setup, "managed_environment_ready",
                        lambda runners_dir, environment=None: False)
    ran = []
    monkeypatch.setattr(android_setup, "setup_android_runners",
                        lambda **kwargs: ran.append(kwargs) or 0)

    said = []
    ready = cli.ensure_target_available(
        "emulator", interactive=True, runners_dir="/runners",
        announce=said.append, prompt=lambda question: "")
    assert ready is True
    assert ran, "setup was never invoked"
    assert "not ready" in "\n".join(said)


def test_optimize_uses_the_same_setup_service_not_a_copy(monkeypatch):
    """One setup implementation, so there is one thing to fix when it breaks."""
    import inspect

    from delegate_doctor import cli

    source = inspect.getsource(cli.ensure_target_available)
    assert "android_setup.setup_android_runners" in source
    assert "sdkmanager" not in source
    assert "avdmanager" not in source


def test_declining_setup_does_not_run_it(monkeypatch):
    from delegate_doctor import cli

    monkeypatch.setattr(android_setup, "managed_environment_ready",
                        lambda runners_dir, environment=None: False)
    ran = []
    monkeypatch.setattr(android_setup, "setup_android_runners",
                        lambda **kwargs: ran.append(kwargs) or 0)

    assert cli.ensure_target_available(
        "emulator", interactive=True, runners_dir="/runners",
        announce=lambda text: None, prompt=lambda question: "n") is False
    assert ran == []


def test_a_ready_environment_is_not_offered_setup(monkeypatch):
    from delegate_doctor import cli

    monkeypatch.setattr(android_setup, "managed_environment_ready",
                        lambda runners_dir, environment=None: True)
    said = []
    assert cli.ensure_target_available(
        "emulator", interactive=True, runners_dir="/runners",
        announce=said.append) is True
    assert said == []


def test_a_device_target_is_never_offered_emulator_setup(monkeypatch):
    """Asking for a phone must not start provisioning an emulator."""
    from delegate_doctor import cli

    called = []
    monkeypatch.setattr(android_setup, "managed_environment_ready",
                        lambda runners_dir, environment=None: called.append(1))

    assert cli.ensure_target_available(
        "device", interactive=True, runners_dir="/runners",
        announce=lambda text: None) is True
    assert called == []


def test_a_non_interactive_run_reports_rather_than_provisions(monkeypatch):
    from delegate_doctor import cli

    monkeypatch.setattr(android_setup, "managed_environment_ready",
                        lambda runners_dir, environment=None: False)
    ran = []
    monkeypatch.setattr(android_setup, "setup_android_runners",
                        lambda **kwargs: ran.append(kwargs) or 0)

    said = []
    assert cli.ensure_target_available(
        "emulator", interactive=False, runners_dir="/runners",
        announce=said.append) is False
    assert ran == []
    assert "setup-android" in "\n".join(said)


# --- Android Studio is the one manual prerequisite -------------------------------

def test_a_standard_android_studio_location_is_discovered(tmp_path, monkeypatch):
    """Case 14: the SDK the Setup Wizard creates is found without configuration."""
    studio = tmp_path / "Library" / "Android" / "sdk"
    (studio / "platform-tools").mkdir(parents=True)
    monkeypatch.setattr(env, "DEFAULT_SDK_LOCATIONS", (str(studio),))

    assert env.find_sdk_root({}, home=tmp_path) == studio


def test_an_sdk_is_validated_by_its_tools_not_its_directory(tmp_path):
    """Case L: a folder surviving an uninstall is not an SDK."""
    empty = tmp_path / "leftover"
    empty.mkdir()
    assert not env.is_usable_sdk(empty)
    (empty / "platform-tools").mkdir()
    assert env.is_usable_sdk(empty)


def test_no_sdk_tells_the_user_to_install_android_studio(tmp_path, monkeypatch):
    """Case 15."""
    monkeypatch.setattr(env, "DEFAULT_SDK_LOCATIONS", ())
    assert env.find_sdk_root({}, home=tmp_path) is None

    message = env.SDK_MISSING_MESSAGE
    assert "Android SDK             NOT FOUND" in message
    assert "Install Android Studio" in message
    assert "initial Setup Wizard" in message
    assert "delegate-doctor setup-android" in message


def test_no_sdk_never_attempts_a_download(tmp_path, monkeypatch):
    """Case 16: setup stops; it does not go looking for an SDK to fetch."""
    environment = make_environment(tmp_path, sdk_root=None)
    said = []
    result = android_setup.ensure_sdk(environment, interactive=True,
                                      assume_yes=True, announce=said.append,
                                      home=tmp_path)
    assert result.sdk_root is None
    printed = "\n".join(said)
    assert "Install Android Studio" in printed
    for banned in ("Downloading", "commandlinetools", "dl.google.com"):
        assert banned not in printed, banned


def test_an_existing_sdk_is_reported_and_reused(tmp_path):
    environment = make_environment(tmp_path, sdk_root=tmp_path / "sdk")
    said = []
    android_setup.ensure_sdk(environment, interactive=True, assume_yes=True,
                             announce=said.append, home=tmp_path)
    printed = "\n".join(said)
    assert "Android Studio          PASS" in printed
    assert "Android SDK             PASS" in printed
    assert str(tmp_path / "sdk") in printed


# --- the bootstrap is gone ----------------------------------------------------------

def test_the_sdk_bootstrap_module_no_longer_exists():
    """Case 31."""
    with pytest.raises(ImportError):
        __import__("delegate_doctor.android_sdk")


def test_no_production_module_can_download_an_sdk():
    """Cases 31/32: nothing reachable fetches or verifies a tools archive."""
    import inspect
    from pathlib import Path

    package = Path(inspect.getfile(android_setup)).parent
    # Symbols, plus the module *import* - not the bare name, which is a
    # substring of the legitimate `check_android_sdk`.
    retired = ("TOOLS_ARCHIVES", "commandlinetools-",
               "bootstrap_command_line_tools", "SdkBootstrapError",
               "unpinned_checksum_message", "managed_sdk_root",
               "from .android_sdk", "import android_sdk",
               "android_sdk.")
    for path in package.rglob("*.py"):
        text = path.read_text()
        for token in retired:
            assert token not in text, f"{path.name} still references {token}"


def test_the_host_is_checked_before_any_system_image_install(tmp_path,
                                                             monkeypatch):
    """Cases 17/18: no multi-gigabyte Arm64 download on a host that cannot use it."""
    installed = []
    monkeypatch.setattr(emulator, "install_packages",
                        lambda environment, packages, announce=print:
                        installed.extend(packages))
    monkeypatch.setattr(emulator, "missing_packages",
                        lambda environment, required=None: list(
                            emulator.EMULATOR_PACKAGES))

    environment = make_environment(tmp_path, sdk_root=tmp_path / "sdk",
                                   system="Linux", machine="x86_64")
    said = []
    ready = android_setup.provision_emulator(environment, interactive=True,
                                             assume_yes=True,
                                             announce=said.append)
    assert ready is False
    assert installed == [], "an Arm64 system image was installed on an x86_64 host"
    assert "UNAVAILABLE" in "\n".join(said)


def test_a_physical_arm64_device_remains_usable_from_any_host(tmp_path,
                                                              monkeypatch):
    """Case 19: the managed emulator is unavailable; a real phone is not."""
    monkeypatch.setattr(android_setup, "runners_already_installed",
                        lambda directory: True)
    environment = make_environment(tmp_path, sdk_root=tmp_path / "sdk",
                                   system="Linux", machine="x86_64")
    verdict = android_setup.format_verdict(tmp_path / "runners", environment)
    assert "physical arm64-v8a Android device" in verdict
    assert "--target device" in verdict
