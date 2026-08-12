"""Suite-wide guarantees: no device, no browser, no network.

The test suite must produce the same result on a laptop with an Arm64 emulator
running as on a CI box with no `adb` at all. Without this, tests silently reach
whatever happens to be plugged in - which is exactly the kind of environmental
dependency that makes a suite untrustworthy.

Tests that want a device mock `pipeline._find_device` themselves; because they
do it inside the test, their patch wins over the autouse fixture here.
"""

import pytest

from delegate_doctor import device, result


@pytest.fixture(autouse=True)
def no_real_android_device(monkeypatch):
    """Device discovery always reports nothing attached."""
    def refuse(*args, **kwargs):
        raise device.DeviceError(
            "No Arm64 Android target is attached.\n(blocked by the test suite)"
        )

    monkeypatch.setattr(device, "require_device", refuse)


@pytest.fixture(autouse=True)
def no_real_browser(monkeypatch):
    """`open_report()` must never actually launch a browser during tests."""
    monkeypatch.setattr(result.webbrowser, "open",
                        lambda *args, **kwargs: pytest.fail(
                            "a test tried to open a real browser"))
