"""Choosing a phone: classification, rejection, and one serial throughout.

DelegateDoctor measures on a physical arm64-v8a Android phone and nothing else.
An emulator is still *detected* - so it can be excluded with a clear message
rather than silently benchmarked - but it is never a usable target, because its
latency describes the host it runs on.

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


def test_an_emulator_is_never_usable_even_when_its_abi_is_right():
    """An arm64 emulator runs the right instructions on the wrong machine."""
    target = an_emulator()
    assert target.is_emulator
    assert target.info.abi == target_selection.REQUIRED_ABI
    assert not target.usable


def test_an_x86_phone_is_not_usable():
    assert not x86_phone().usable


def test_usable_targets_keeps_only_phones():
    targets = [an_emulator(), phone(), x86_phone()]
    usable = target_selection.usable_targets(targets)
    assert [t.serial for t in usable] == ["RMX-123"]


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


def test_an_emulator_is_never_selected_even_when_it_is_the_only_target():
    with pytest.raises(DeviceError) as caught:
        target_selection.select_target(targets=[an_emulator()],
                                       announce=lambda *a: None)
    message = str(caught.value)
    assert "arm64-v8a" in message


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
