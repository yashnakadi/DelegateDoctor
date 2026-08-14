"""Choosing a target: classification, preference, and one serial throughout.

The policy under test:

    A physical arm64-v8a Android phone is the supported and validated target
    and always wins. An already-running arm64-v8a emulator is a best-effort
    fallback - usable when no phone is, never preferred, and always warned
    about.

An x86_64 target is refused either way: it says nothing about Arm.

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


def an_emulator(serial="emulator-5554"):
    return make_target(serial=serial, model="sdk_gphone64_arm64",
                       release="15", hardware="ranchu",
                       manufacturer="Google", avd_name="DelegateDoctor_ARM64")


def x86_phone(serial="tablet-1"):
    return make_target(serial=serial, model="SomeTablet", abi="x86_64",
                       hardware="intel", manufacturer="Acme")


# --- classification ----------------------------------------------------------

def test_a_physical_arm64_phone_is_usable():
    assert phone().usable


def test_an_arm64_emulator_is_usable_as_a_best_effort_target():
    """Usable, but never preferred - see the preference tests below."""
    target = an_emulator()
    assert target.is_emulator
    assert target.info.abi == target_selection.REQUIRED_ABI
    assert target.usable


def test_an_x86_emulator_is_not_usable():
    x86 = make_target(serial="e", model="sdk_gphone64_x86_64", abi="x86_64",
                      hardware="ranchu")
    assert not x86.usable


def test_an_x86_phone_is_not_usable():
    assert not x86_phone().usable


def test_usable_targets_puts_phones_first():
    """Ordering is the whole preference mechanism."""
    targets = [an_emulator(), phone(), x86_phone()]
    usable = target_selection.usable_targets(targets)
    assert [t.serial for t in usable] == ["RMX-123", "emulator-5554"]
    assert [t.serial for t in target_selection.physical_targets(targets)] == \
        ["RMX-123"]


def test_the_short_description_names_the_phone_and_its_abi():
    text = phone().short_description()
    assert "RMX2030" in text and "arm64-v8a" in text


# --- selection ---------------------------------------------------------------

def test_one_phone_is_selected_automatically():
    chosen = target_selection.select_target(targets=[phone()],
                                            announce=lambda *a: None)
    assert chosen.serial == "RMX-123"


def test_an_explicit_serial_wins():
    targets = [phone("A"), phone("B")]
    chosen = target_selection.select_target(serial="B", targets=targets,
                                            announce=lambda *a: None)
    assert chosen.serial == "B"


def test_an_explicit_serial_that_is_not_attached_is_refused():
    with pytest.raises(DeviceError) as caught:
        target_selection.select_target(serial="ghost", targets=[phone("A")],
                                       announce=lambda *a: None)
    assert "ghost" in str(caught.value)


def test_an_explicit_x86_serial_is_refused():
    with pytest.raises(DeviceError) as caught:
        target_selection.select_target(serial="tablet-1",
                                       targets=[x86_phone()],
                                       announce=lambda *a: None)
    assert "arm64-v8a" in str(caught.value)


def test_several_phones_produce_a_question():
    asked = []
    chosen = target_selection.select_target(
        targets=[phone("A"), phone("B")], interactive=True,
        prompt=lambda text: asked.append(text) or "2",
        announce=lambda *a: None)
    assert chosen.serial == "B"
    assert asked, "the user was never asked which phone to use"


def test_the_default_answer_selects_the_first():
    chosen = target_selection.select_target(
        targets=[phone("A"), phone("B")], interactive=True,
        prompt=lambda text: "", announce=lambda *a: None)
    assert chosen.serial == "A"


def test_an_invalid_answer_is_re_asked():
    answers = iter(["nonsense", "9", "1"])
    chosen = target_selection.select_target(
        targets=[phone("A"), phone("B")], interactive=True,
        prompt=lambda text: next(answers), announce=lambda *a: None)
    assert chosen.serial == "A"


def test_non_interactive_selection_never_prompts():
    chosen = target_selection.select_target(
        targets=[phone("A"), phone("B")], interactive=False,
        prompt=lambda text: pytest.fail("prompted in a non-interactive run"),
        announce=lambda *a: None)
    assert chosen.serial == "A"


def test_an_emulator_alone_is_used_with_a_warning():
    """Best-effort rather than a refusal, but the user is told."""
    said = []
    chosen = target_selection.select_target(targets=[an_emulator()],
                                            announce=said.append)
    assert chosen.is_emulator
    text = "\n".join(said)
    assert "Arm64 Android emulator" in text
    assert "not a validated" in text


def test_a_phone_wins_over_an_emulator_without_asking():
    """A running emulator must not turn a one-phone machine into a question."""
    chosen = target_selection.select_target(
        targets=[an_emulator(), phone()], interactive=True,
        prompt=lambda text: pytest.fail("asked despite an unambiguous phone"),
        announce=lambda *a: None)
    assert chosen.serial == "RMX-123"
    assert not chosen.is_emulator


def test_a_phone_run_carries_no_emulator_warning():
    said = []
    target_selection.select_target(targets=[phone()], announce=said.append)
    assert "emulator" not in "\n".join(said).lower()


def test_several_emulators_and_no_phone_require_a_choice():
    said = []
    chosen = target_selection.select_target(
        targets=[an_emulator("emulator-5554"), an_emulator("emulator-5556")],
        interactive=False, announce=said.append)
    assert chosen.is_emulator
    assert "--device" in "\n".join(said)


def test_an_unauthorized_phone_does_not_block_a_usable_emulator():
    """discover_targets drops non-`device` states, so only the emulator remains."""
    chosen = target_selection.select_target(targets=[an_emulator()],
                                            announce=lambda *a: None)
    assert chosen.is_emulator


def test_device_selects_an_emulator_by_serial_with_the_warning():
    said = []
    chosen = target_selection.select_target(
        serial="emulator-5554", targets=[phone(), an_emulator()],
        announce=said.append)
    assert chosen.serial == "emulator-5554"
    assert "not a validated" in "\n".join(said)


def test_device_refuses_an_x86_emulator_by_serial():
    x86 = make_target(serial="emulator-9", model="sdk_gphone64_x86_64",
                      abi="x86_64", hardware="ranchu")
    with pytest.raises(DeviceError) as caught:
        target_selection.select_target(serial="emulator-9", targets=[x86],
                                       announce=lambda *a: None)
    message = str(caught.value)
    assert "x86_64" in message and "arm64-v8a" in message


def test_no_targets_at_all_explains_how_to_get_one():
    with pytest.raises(DeviceError) as caught:
        target_selection.select_target(targets=[], announce=lambda *a: None)
    message = str(caught.value)
    assert "USB debugging" in message
    assert "phone" in message.lower()


# --- one serial, carried everywhere ------------------------------------------

def test_the_selected_serial_is_what_the_pipeline_uses(monkeypatch):
    monkeypatch.setattr(target_selection, "discover_targets",
                        lambda: [phone("CHOSEN")])
    monkeypatch.setattr(pipeline.device, "find_runner",
                        lambda runners_dir, name: f"/runners/{name}")
    info, bench, etdump, error = pipeline._find_device("/runners")
    assert error == ""
    assert info.serial == "CHOSEN"


def test_the_pipeline_reports_no_device_instead_of_raising(monkeypatch):
    monkeypatch.setattr(target_selection, "discover_targets", lambda: [])
    info, bench, etdump, error = pipeline._find_device("/runners")
    assert info is None
    assert error


# --- the public CLI has one target concept -----------------------------------

def test_the_cli_has_no_target_option(capsys):
    """`--target` is gone: there is only one kind of target now."""
    from delegate_doctor import cli

    with pytest.raises(SystemExit):
        cli.main(["optimize", "--help"])
    text = capsys.readouterr().out
    assert "--target" not in text
    assert "--device" in text, "selecting among several phones must remain"


def test_the_cli_rejects_a_target_flag():
    from delegate_doctor import cli

    with pytest.raises(SystemExit):
        cli.main(["optimize", "model.py", "--target", "device"])


def test_the_model_path_is_not_overwritten_by_the_device_flag(monkeypatch):
    from delegate_doctor import cli

    seen = {}
    monkeypatch.setattr(cli, "run_optimize",
                        lambda target, **options: seen.update(
                            {"target": target, **options}) or 0)
    assert cli.main(["optimize", "m.py", "--device", "SERIAL-9"]) == 0
    assert seen["target"] == "m.py"
    assert seen["target_serial"] == "SERIAL-9"


# --- discovery reads adb once ------------------------------------------------

def test_discovery_skips_targets_that_are_not_ready(monkeypatch):
    monkeypatch.setattr(target_selection, "list_adb_serials",
                        lambda: [("A", "device"), ("B", "unauthorized"),
                                 ("C", "offline")])
    probed = []
    monkeypatch.setattr(target_selection, "probe_target",
                        lambda serial: probed.append(serial) or phone(serial))
    targets = target_selection.discover_targets()
    assert [t.serial for t in targets] == ["A"]
    assert probed == ["A"], "an unusable target was probed anyway"
