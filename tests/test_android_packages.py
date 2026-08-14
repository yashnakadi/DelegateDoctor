"""Android SDK package inspection, installation and licences.

What used to be `test_emulator.py`. DelegateDoctor is physical-phone-only now,
so the AVD creation, boot and launch tests went with the feature; what remains
is the SDK package management `setup-android` still needs.

Fully offline: `run_tool` is stubbed everywhere, so no sdkmanager ever runs.
"""

import subprocess

import pytest

from delegate_doctor import android_environment as env
from delegate_doctor import android_packages
from delegate_doctor.android_environment import AndroidEnvironment, AndroidTool
from delegate_doctor.android_packages import AndroidPackageError, ToolResult
# Captured at import time: conftest deliberately blocks the module attribute.


def make_environment(tmp_path, **overrides):
    """An environment whose tools all resolve to plausible paths."""
    tools = {
        name: AndroidTool(name, tmp_path / name)
        for name in ("adb", "sdkmanager")
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



def test_the_required_packages_are_explicit():
    """Two entries: adb to reach the phone, the pinned NDK to build runners.

    Nothing else. There is no emulator package, no SDK platform and no system
    image, because DelegateDoctor measures on a real phone.
    """
    assert android_packages.REQUIRED_PACKAGES == (
        "platform-tools", android_packages.NDK_PACKAGE,
    )
    for gone in ("emulator", "platforms;android-35",
                 "system-images;android-35;google_apis;arm64-v8a"):
        assert gone not in android_packages.REQUIRED_PACKAGES


def test_the_ndk_is_pinned_not_latest():
    assert android_packages.NDK_PACKAGE == "ndk;27.2.12479018"
    assert "latest" not in android_packages.NDK_PACKAGE



def test_the_command_timeout_is_a_named_bounded_constant():
    assert isinstance(android_packages.COMMAND_TIMEOUT_SECONDS, int)
    assert 0 < android_packages.COMMAND_TIMEOUT_SECONDS <= 600


INSTALLED_OUTPUT = """Installed packages:
  Path                 | Version | Description       | Location
  -------              | ------- | -------           | -------
  emulator             | 35.1.4  | Android Emulator  | android_packages
  platform-tools       | 35.0.2  | Android SDK Platform-Tools | platform-tools
  platforms;android-35 | 2       | Android SDK Platform 35 | platforms/android-35
"""


def test_installed_packages_are_parsed():
    packages = android_packages.parse_installed_packages(INSTALLED_OUTPUT)
    assert "emulator" in packages
    assert "platform-tools" in packages
    assert "platforms;android-35" in packages


def test_available_packages_are_not_counted_as_installed():
    text = INSTALLED_OUTPUT + """
Available Packages:
  Path                                            | Version
  system-images;android-35;google_apis;arm64-v8a  | 1
"""
    packages = android_packages.parse_installed_packages(text)
    assert "system-images;android-35;google_apis;arm64-v8a" not in packages


def test_nothing_missing_when_everything_is_installed(tmp_path, monkeypatch):
    complete = INSTALLED_OUTPUT + \
        f"  {android_packages.NDK_PACKAGE} | 1 | NDK | ndk\n"
    monkeypatch.setattr(android_packages, "run_tool",
                        recorder({"--list_installed": ToolResult(0, complete, "")}))
    assert android_packages.missing_packages(make_environment(tmp_path)) == []


def test_one_missing_package_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(android_packages, "run_tool",
                        recorder({"--list_installed":
                                  ToolResult(0, INSTALLED_OUTPUT, "")}))
    missing = android_packages.missing_packages(make_environment(tmp_path))
    assert missing == [android_packages.NDK_PACKAGE]


def test_several_missing_packages_keep_their_declared_order(tmp_path, monkeypatch):
    monkeypatch.setattr(android_packages, "run_tool",
                        recorder({"--list_installed":
                                  ToolResult(0, "Installed packages:\n", "")}))
    missing = android_packages.missing_packages(make_environment(tmp_path))
    assert missing == list(android_packages.REQUIRED_PACKAGES)


def test_install_names_exactly_the_missing_packages(tmp_path, monkeypatch):
    fake = recorder()
    monkeypatch.setattr(android_packages, "run_tool", fake)
    android_packages.install_packages(make_environment(tmp_path),
                              ["emulator", "platforms;android-35"],
                              announce=lambda *a: None)
    installed = [call["arguments"] for call in fake.calls]
    assert installed == [["--install", "emulator"],
                         ["--install", "platforms;android-35"]]


def test_installing_nothing_runs_nothing(tmp_path, monkeypatch):
    fake = recorder()
    monkeypatch.setattr(android_packages, "run_tool", fake)
    android_packages.install_packages(make_environment(tmp_path), [])
    assert fake.calls == []


def test_a_failed_install_mentions_licences(tmp_path, monkeypatch):
    monkeypatch.setattr(android_packages, "run_tool",
                        recorder({"--install": ToolResult(1, "", "not accepted")}))
    with pytest.raises(AndroidPackageError) as caught:
        android_packages.install_packages(make_environment(tmp_path), ["emulator"],
                                  announce=lambda *a: None)
    assert "--licenses" in str(caught.value)


# --- licences ----------------------------------------------------------------

def test_licences_are_detected_when_accepted(tmp_path):
    sdk = tmp_path / "sdk"
    (sdk / "licenses").mkdir(parents=True)
    (sdk / "licenses" / "android-sdk-license").write_text("hash")
    assert android_packages.licenses_accepted(make_environment(tmp_path, sdk_root=sdk))


def test_missing_licences_are_detected(tmp_path):
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    assert not android_packages.licenses_accepted(make_environment(tmp_path, sdk_root=sdk))


def test_delegate_doctor_does_not_accept_licences_on_the_users_behalf():
    """No `yes |` piping: the user accepts Google's terms, not this tool."""
    source = _code_without_prose(android_packages.__file__)
    for pattern in ("yes |", '"y\\n" * ', "--licenses\"], input", "accept_licenses("):
        assert pattern not in source, f"android_packages.py auto-accepts licences: {pattern}"
    assert "sdkmanager --licenses" in android_packages.LICENSES_MESSAGE


# --- AVD ---------------------------------------------------------------------
