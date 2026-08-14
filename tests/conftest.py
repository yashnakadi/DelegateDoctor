"""Suite-wide guarantees: no device, no browser, no network.

The test suite must produce the same result on a laptop with an Arm64 emulator
running as on a CI box with no `adb` at all. Without this, tests silently reach
whatever happens to be plugged in - which is exactly the kind of environmental
dependency that makes a suite untrustworthy.

Tests that want a device mock `pipeline._find_device` themselves; because they
do it inside the test, their patch wins over the autouse fixture here.
"""

import pytest

from delegate_doctor import device, emulator, result, target_selection
from delegate_doctor.agent import client as ai_client


@pytest.fixture(autouse=True)
def no_real_android_device(monkeypatch):
    """Device discovery always reports nothing attached.

    Every route to a real target is blocked, not just the one the pipeline
    happens to use today: `run_adb` is the single choke point through which any
    adb call must pass, so blocking it makes the guard hard to outgrow.
    """
    def refuse(*args, **kwargs):
        raise device.DeviceError(
            "No Arm64 Android target is attached.\n(blocked by the test suite)"
        )

    # `run_adb` is the single choke point every adb call passes through, so
    # blocking it covers discovery, probing and selection at once - and leaves
    # individual tests free to patch it, or `discover_targets`, themselves.
    monkeypatch.setattr(device, "require_device", refuse)
    monkeypatch.setattr(device, "run_adb", refuse)
    monkeypatch.setattr(target_selection, "run_adb", refuse)


@pytest.fixture(autouse=True)
def no_real_android_tooling(monkeypatch):
    """No test may run sdkmanager, avdmanager or the emulator.

    `run_tool` is the single choke point for every Android SDK command, and
    `launch_emulator_process` is the only way an emulator is ever started, so
    blocking both means a stray test cannot download an SDK package, create an
    AVD, or boot a virtual device.
    """
    def refuse_tool(*args, **kwargs):
        raise emulator.EmulatorError("Android tooling is blocked by the test suite")

    monkeypatch.setattr(emulator, "run_tool", refuse_tool)
    monkeypatch.setattr(emulator, "launch_emulator_process",
                        lambda *args, **kwargs: pytest.fail(
                            "a test tried to start a real emulator"))


@pytest.fixture(autouse=True)
def no_real_ai_provider(monkeypatch):
    """No test may contact an AI provider, spend credits, or send source.

    A developer running pytest with a real API key configured must not have
    DelegateDoctor phone home because of a regression. Both the transport and
    the factory are blocked: tests inject a fake provider instead.
    """
    def refuse_completion(*args, **kwargs):
        pytest.fail("a test tried to make a real LiteLLM provider request")

    def refuse_build(*args, **kwargs):
        raise ai_client.AINotConfigured(
            "provider construction is blocked by the test suite")

    # LiteLLM's own entry point is the single choke point for every provider,
    # so blocking it stops OpenAI, Anthropic, Gemini, OpenRouter and Ollama
    # requests at once - even for a developer with real keys exported.
    try:
        import litellm

        monkeypatch.setattr(litellm, "completion", refuse_completion)
    except ImportError:
        pass

    monkeypatch.setattr(ai_client, "build_provider", refuse_build)


@pytest.fixture(autouse=True)
def no_real_browser(monkeypatch):
    """`open_report()` must never actually launch a browser during tests."""
    monkeypatch.setattr(result.webbrowser, "open",
                        lambda *args, **kwargs: pytest.fail(
                            "a test tried to open a real browser"))


@pytest.fixture(autouse=True)
def healthy_environment(monkeypatch):
    """`run_optimize` preflights the environment; unit tests are not about that.

    Without this the whole CLI suite would pass or fail according to whether
    the *developer's* machine has a working pandas, which is exactly the kind
    of hidden coupling the preflight exists to expose in production and has no
    business having in tests. Tests that are about the preflight replace it
    themselves.
    """
    from delegate_doctor import environment_check

    monkeypatch.setattr(environment_check, "preflight",
                        lambda *args, **kwargs: environment_check.EnvironmentReport())
