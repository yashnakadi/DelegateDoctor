"""Choosing an Arm target: classification, rejection, and one serial throughout.

Offline. `adb` is mocked at the `run_adb` boundary, so nothing here touches a
real device - which is also the point of several of the tests.
"""

import pytest

from delegate_doctor import device, pipeline, target_selection
from delegate_doctor.device import DeviceError, DeviceInfo
from delegate_doctor.target_selection import Target


# --- fixtures ----------------------------------------------------------------

def make_target(serial="serial-1", model="RMX2030", abi="arm64-v8a",
                release="10", hardware="qcom", manufacturer="realme",
                avd_name=""):
    return Target(
        info=DeviceInfo(serial=serial, model=model, abi=abi,
                        android_release=release, sdk_level="29",
                        hardware=hardware),
        manufacturer=manufacturer,
        avd_name=avd_name,
    )


def phone(serial="RMX-123"):
    return make_target(serial=serial)


def emulator(serial="emulator-5554"):
    return make_target(serial=serial, model="sdk_gphone64_arm64",
                       release="15", hardware="ranchu",
                       manufacturer="Google", avd_name="DelegateDoctor_ARM64")


def x86_emulator(serial="emulator-5556"):
    return make_target(serial=serial, model="sdk_gphone64_x86_64",
                       abi="x86_64", release="15", hardware="ranchu")


def silent(*args, **kwargs):
    """An `announce` that prints nothing, for non-UX assertions."""


# --- classification ----------------------------------------------------------

def test_an_emulator_is_classified_as_an_emulator():
    target = emulator()
    assert target.is_emulator
    assert target.kind == "emulator"
    assert target.kind_label == "Android Emulator"


def test_a_physical_device_is_classified_as_physical():
    target = phone()
    assert not target.is_emulator
    assert target.kind == "physical"
    assert target.kind_label == "Physical Android Device"


def test_an_emulator_is_named_by_its_avd():
    assert emulator().display_name == "DelegateDoctor_ARM64"


def test_a_phone_is_named_by_manufacturer_and_model():
    assert make_target(model="Pixel 8", manufacturer="Google").display_name == \
        "Google Pixel 8"


def test_a_redundant_manufacturer_is_not_repeated():
    assert make_target(model="realme RMX2030",
                       manufacturer="realme").display_name == "realme RMX2030"


def test_an_arm64_target_is_usable_and_x86_is_not():
    assert phone().usable
    assert emulator().usable
    assert not x86_emulator().usable


def test_the_short_description_states_the_kind():
    assert "Physical" in phone().short_description()
    assert "Emulator" in emulator().short_description()


# --- discovery ---------------------------------------------------------------

def test_only_online_targets_are_probed(monkeypatch):
    listing = ("List of devices attached\n"
               "serial-online\tdevice\n"
               "serial-unauth\tunauthorized\n"
               "serial-offline\toffline\n")
    probed = []

    monkeypatch.setattr(target_selection, "run_adb", lambda *a, **k: listing)
    monkeypatch.setattr(target_selection, "probe_target",
                        lambda serial: probed.append(serial) or phone(serial))

    targets = target_selection.discover_targets()
    assert probed == ["serial-online"]
    assert len(targets) == 1


def test_an_empty_device_list_discovers_nothing(monkeypatch):
    monkeypatch.setattr(target_selection, "run_adb",
                        lambda *a, **k: "List of devices attached\n\n")
    assert target_selection.discover_targets() == []


def test_metadata_is_read_from_getprop(monkeypatch):
    answers = {
        "ro.product.model": "RMX2030",
        "ro.product.cpu.abi": "arm64-v8a",
        "ro.build.version.release": "10",
        "ro.build.version.sdk": "29",
        "ro.hardware": "qcom",
        "ro.product.manufacturer": "realme",
    }

    def fake_adb(*args, serial=None, **kwargs):
        return answers.get(args[-1], "")

    monkeypatch.setattr(target_selection, "run_adb", fake_adb)
    target = target_selection.probe_target("RMX-1")
    assert target.info.model == "RMX2030"
    assert target.info.abi == "arm64-v8a"
    assert target.manufacturer == "realme"
    assert target.serial == "RMX-1"


# --- selection ---------------------------------------------------------------

def test_a_single_target_is_used_without_asking():
    asked = []
    chosen = target_selection.select_target(
        targets=[phone()], prompt=lambda _: asked.append(1),
        announce=silent)
    assert chosen.serial == "RMX-123"
    assert asked == []


def test_the_single_target_is_still_announced():
    lines = []
    target_selection.select_target(targets=[phone()], announce=lines.append,
                                   prompt=lambda _: "")
    assert any("RMX2030" in line for line in lines)


def test_several_targets_produce_a_question():
    answers = iter(["2"])
    chosen = target_selection.select_target(
        targets=[phone(), emulator()],
        prompt=lambda _: next(answers), announce=silent)
    assert chosen.is_emulator


def test_the_default_answer_selects_the_first():
    chosen = target_selection.select_target(
        targets=[phone(), emulator()], prompt=lambda _: "", announce=silent)
    assert chosen.serial == "RMX-123"


def test_a_physical_device_is_offered_before_an_emulator():
    ordered = target_selection.usable_targets([emulator(), phone()])
    assert not ordered[0].is_emulator


def test_an_invalid_answer_is_re_asked():
    answers = iter(["nonsense", "9", "1"])
    messages = []
    chosen = target_selection.select_target(
        targets=[phone(), emulator()],
        prompt=lambda _: next(answers), announce=messages.append)
    assert chosen.serial == "RMX-123"
    assert any("between 1 and 2" in str(m) for m in messages)


def test_non_interactive_selection_never_prompts():
    chosen = target_selection.select_target(
        targets=[phone(), emulator()], interactive=False,
        prompt=lambda _: pytest.fail("must not prompt"), announce=silent)
    assert not chosen.is_emulator


# --- ABI is enforced before anything is measured -----------------------------

def test_an_x86_emulator_is_never_selected():
    """It runs Android fine and says nothing about Arm."""
    with pytest.raises(DeviceError) as caught:
        target_selection.select_target(targets=[x86_emulator()], announce=silent)
    message = str(caught.value)
    assert "ARM TARGET NOT FOUND" in message
    assert "x86_64" in message
    assert "says nothing about Arm performance" in message


def test_an_x86_target_is_skipped_when_an_arm_one_exists():
    chosen = target_selection.select_target(
        targets=[x86_emulator(), phone()], announce=silent,
        prompt=lambda _: pytest.fail("only one usable target; should not ask"))
    assert chosen.info.abi == "arm64-v8a"


def test_an_explicit_x86_serial_is_refused():
    with pytest.raises(DeviceError) as caught:
        target_selection.select_target(serial="emulator-5556",
                                       targets=[x86_emulator(), phone()],
                                       announce=silent)
    assert "requires arm64-v8a" in str(caught.value)


def test_no_targets_at_all_explains_how_to_get_one():
    with pytest.raises(DeviceError) as caught:
        target_selection.select_target(targets=[], announce=silent)
    message = str(caught.value)
    assert "ARM TARGET NOT FOUND" in message
    assert "adb devices" in message
    assert "setup-android" in message


# --- explicit selection ------------------------------------------------------

def test_an_explicit_serial_wins():
    chosen = target_selection.select_target(
        serial="emulator-5554", targets=[phone(), emulator()],
        prompt=lambda _: pytest.fail("must not ask"), announce=silent)
    assert chosen.serial == "emulator-5554"


def test_an_unknown_serial_lists_what_is_attached():
    with pytest.raises(DeviceError) as caught:
        target_selection.select_target(serial="nope", targets=[phone()],
                                       announce=silent)
    assert "RMX-123" in str(caught.value)


def test_the_emulator_preference_selects_the_emulator():
    chosen = target_selection.select_target(
        preference="emulator", targets=[phone(), emulator()], announce=silent)
    assert chosen.is_emulator


def test_the_device_preference_selects_the_phone():
    chosen = target_selection.select_target(
        preference="device", targets=[phone(), emulator()], announce=silent)
    assert not chosen.is_emulator


def test_the_emulator_preference_fails_clearly_when_none_is_running():
    with pytest.raises(DeviceError) as caught:
        target_selection.select_target(preference="emulator", targets=[phone()],
                                       announce=silent)
    assert "No arm64-v8a Android emulator is running" in str(caught.value)


def test_the_device_preference_fails_clearly_when_none_is_connected():
    with pytest.raises(DeviceError) as caught:
        target_selection.select_target(preference="device", targets=[emulator()],
                                       announce=silent)
    assert "No physical arm64-v8a Android device" in str(caught.value)


def test_an_unknown_preference_is_refused():
    with pytest.raises(DeviceError) as caught:
        target_selection.select_target(preference="laptop", targets=[phone()])
    assert "Unknown target preference" in str(caught.value)


# --- one serial reaches every device stage -----------------------------------

def test_the_selected_serial_is_what_the_pipeline_uses(monkeypatch):
    monkeypatch.setattr(target_selection, "discover_targets",
                        lambda: [phone("CHOSEN-1"), emulator("OTHER-2")])
    monkeypatch.setattr(pipeline.device, "find_runner",
                        lambda runners_dir, name: f"/runners/{name}")

    info, bench, etdump, reason = pipeline._find_device(
        "/runners", target_preference="device")
    assert info.serial == "CHOSEN-1"
    assert reason == ""


def test_the_pipeline_reports_no_device_instead_of_raising(monkeypatch):
    monkeypatch.setattr(target_selection, "discover_targets", lambda: [])
    info, bench, etdump, reason = pipeline._find_device("/runners")
    assert info is None
    assert "ARM TARGET NOT FOUND" in reason


def test_a_missing_runner_is_also_a_capability_not_a_crash(monkeypatch):
    monkeypatch.setattr(target_selection, "discover_targets", lambda: [phone()])
    info, bench, etdump, reason = pipeline._find_device("/nonexistent-runners")
    assert info is None
    assert reason


def test_the_pipeline_never_prompts_by_default(monkeypatch):
    """A library call must not block on stdin."""
    monkeypatch.setattr(target_selection, "discover_targets",
                        lambda: [phone("A"), emulator("B")])
    monkeypatch.setattr(pipeline.device, "find_runner",
                        lambda runners_dir, name: name)
    monkeypatch.setattr("builtins.input",
                        lambda *args: pytest.fail("the pipeline prompted"))
    info, _, _, _ = pipeline._find_device("/runners")
    assert info.serial == "A"


# --- the CLI surface ---------------------------------------------------------

def test_the_cli_exposes_target_selection_flags():
    from delegate_doctor import cli

    seen = {}
    original = cli.run_optimize
    try:
        cli.run_optimize = lambda target, **options: (
            seen.update(model=target, **options) or 0)
        cli.main(["optimize", "m.py",
                  "--target", "emulator", "--device", "SER-9",
                  "--non-interactive"])
    finally:
        cli.run_optimize = original

    assert seen["model"] == "m.py"           # the positional survives --target
    assert seen["target_preference"] == "emulator"
    assert seen["target_serial"] == "SER-9"
    assert seen["interactive"] is False


def test_the_model_path_is_not_overwritten_by_the_target_flag():
    """argparse would silently collide these without an explicit dest."""
    from delegate_doctor import cli

    seen = {}
    original = cli.run_optimize
    try:
        cli.run_optimize = lambda target, **options: (
            seen.update(model=target) or 0)
        cli.main(["optimize", "models/private.py", "--target", "device"])
    finally:
        cli.run_optimize = original
    assert seen["model"] == "models/private.py"


# --- Phase 3 integration: the managed emulator -------------------------------

def test_a_running_managed_emulator_is_discovered_normally():
    """Already running: nothing special happens, it is just a target."""
    chosen = target_selection.select_target(
        preference="emulator", targets=[emulator("emulator-5554")],
        announce=silent)
    assert chosen.display_name == "DelegateDoctor_ARM64"


def test_an_unsupported_host_says_so_rather_than_offering_an_emulator(monkeypatch):
    """x86_64 host: the answer is a physical device, not an x86 image."""
    from delegate_doctor import android_environment

    monkeypatch.setattr(
        android_environment, "detect",
        lambda **kwargs: android_environment.AndroidEnvironment(
            host=android_environment.detect_host(system="Linux", machine="x86_64")))

    with pytest.raises(DeviceError) as caught:
        target_selection.select_target(preference="emulator", targets=[],
                                       announce=silent)
    message = str(caught.value)
    assert "ARM64 EMULATOR UNAVAILABLE ON THIS HOST" in message
    assert "physical arm64-v8a" in message


def test_no_emulator_is_started_when_the_avd_does_not_exist(monkeypatch):
    from delegate_doctor import android_environment, emulator as emulator_module

    monkeypatch.setattr(
        android_environment, "detect",
        lambda **kwargs: android_environment.AndroidEnvironment(
            host=android_environment.detect_host(system="Darwin", machine="arm64")))
    monkeypatch.setattr(emulator_module, "avd_exists", lambda e: False)
    monkeypatch.setattr(
        emulator_module, "start_delegate_doctor_emulator",
        lambda *a, **k: pytest.fail("must not start a nonexistent AVD"))

    with pytest.raises(DeviceError) as caught:
        target_selection.select_target(preference="emulator", targets=[],
                                       announce=silent)
    assert "setup-android" in str(caught.value)


def test_optimize_never_provisions_only_starts(monkeypatch):
    """Provisioning belongs to setup-android; optimize may only start."""
    import inspect

    source = inspect.getsource(target_selection._offer_managed_emulator)
    for provisioning in ("install_packages", "create_avd", "missing_packages"):
        assert provisioning not in source, (
            f"target selection performs provisioning: {provisioning}")


def test_a_started_emulator_becomes_the_selected_target(monkeypatch):
    from delegate_doctor import android_environment, emulator as emulator_module

    environment = android_environment.AndroidEnvironment(
        host=android_environment.detect_host(system="Darwin", machine="arm64"),
        sdk_root="/sdk",
        tools={name: android_environment.AndroidTool(name, f"/sdk/{name}")
               for name in ("adb", "emulator", "sdkmanager", "avdmanager")},
    )
    monkeypatch.setattr(android_environment, "detect", lambda **kwargs: environment)
    monkeypatch.setattr(emulator_module, "avd_exists", lambda e: True)
    monkeypatch.setattr(emulator_module, "start_delegate_doctor_emulator",
                        lambda e, announce=None: "emulator-5588")
    monkeypatch.setattr(target_selection, "discover_targets",
                        lambda: [emulator("emulator-5588")])

    chosen = target_selection.select_target(
        preference="emulator", targets=[], interactive=False, announce=silent)
    assert chosen.serial == "emulator-5588"


def test_a_declined_prompt_does_not_start_anything(monkeypatch):
    from delegate_doctor import android_environment, emulator as emulator_module

    monkeypatch.setattr(
        android_environment, "detect",
        lambda **kwargs: android_environment.AndroidEnvironment(
            host=android_environment.detect_host(system="Darwin", machine="arm64"),
            sdk_root="/sdk",
            tools={name: android_environment.AndroidTool(name, f"/sdk/{name}")
                   for name in ("adb", "emulator", "sdkmanager", "avdmanager")}))
    monkeypatch.setattr(emulator_module, "avd_exists", lambda e: True)
    monkeypatch.setattr(
        emulator_module, "start_delegate_doctor_emulator",
        lambda *a, **k: pytest.fail("the user said no"))

    with pytest.raises(DeviceError):
        target_selection.select_target(preference="emulator", targets=[],
                                       interactive=True, prompt=lambda _: "n",
                                       announce=silent)


def test_a_physical_device_run_never_consults_the_emulator(monkeypatch):
    """Unchanged Phase 2 behaviour: a phone is selected without any SDK work."""
    from delegate_doctor import emulator as emulator_module

    monkeypatch.setattr(emulator_module, "avd_exists",
                        lambda e: pytest.fail("emulator machinery was consulted"))
    chosen = target_selection.select_target(preference="device",
                                            targets=[phone()], announce=silent)
    assert not chosen.is_emulator
