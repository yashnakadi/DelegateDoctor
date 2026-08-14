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
from delegate_doctor import android_packages, android_setup
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


def make_environment(tmp_path, sdk_root=None, tools=("adb",
                                                     "sdkmanager", "avdmanager"),
                     system="Darwin", machine="arm64"):
    return AndroidEnvironment(
        host=env.detect_host(system=system, machine=machine),
        sdk_root=sdk_root,
        tools={name: AndroidTool(name, tmp_path / name) for name in tools},
    )










# --- licences ------------------------------------------------------------------------


def test_unaccepted_licences_stop_a_non_interactive_run(tmp_path, monkeypatch):
    """No flag synthesizes agreement, and none is offered."""
    monkeypatch.setattr(android_packages, "licenses_accepted", lambda environment: False)
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

    source = inspect.getsource(android_packages)
    # Piping agreement into the licence prompt is the specific thing forbidden.
    assert 'input_text="y' not in source
    assert '"yes\\n" * ' not in source


def test_the_licence_process_gets_the_users_terminal(tmp_path, monkeypatch):
    """Not captured: a user cannot agree to text they were never shown."""
    monkeypatch.setattr(android_packages, "licenses_accepted", lambda environment: True)
    commands = []
    android_packages.accept_licenses_interactively(
        make_environment(tmp_path, sdk_root=tmp_path / "sdk"),
        announce=lambda text: None,
        prompt=lambda question: "",
        runner=lambda command: commands.append(command) or 0)
    assert commands and commands[0][-1] == "--licenses"


# --- AVD safety -----------------------------------------------------------------------










# --- host policy --------------------------------------------------------------------










# --- optimize can trigger setup --------------------------------------------------------




def test_optimize_uses_the_same_setup_service_not_a_copy(monkeypatch):
    """One setup implementation, so there is one thing to fix when it breaks."""
    import inspect

    from delegate_doctor import cli

    source = inspect.getsource(cli.ensure_target_available)
    assert "android_setup.setup_android_runners" in source
    assert "sdkmanager" not in source
    assert "avdmanager" not in source










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
