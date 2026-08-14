"""Find the Arm targets attached to this machine, and choose one deliberately.

The old behaviour was "exactly one device must be attached, or fail". That is
fine for a single phone on a desk and useless the moment an emulator is also
running. This module discovers every adb target, describes it honestly, and
picks one - by explicit selection, by a stated preference, or by asking.

Two things matter more than convenience here:

  * **The ABI is checked before anything is measured.** An x86_64 emulator can
    run an Android app perfectly well and tell you nothing about Arm, so it is
    never offered as a performance target.

  * **One serial is chosen once and carried everywhere.** Profiling, device
    verification and the benchmark all receive it explicitly, so a second
    device appearing mid-run cannot split a before/after comparison across two
    machines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .device import DeviceError, DeviceInfo, run_adb

REQUIRED_ABI = "arm64-v8a"

# What `--target` accepts.
PREFERENCE_AUTO = "auto"
PREFERENCE_EMULATOR = "emulator"
PREFERENCE_DEVICE = "device"
PREFERENCES = (PREFERENCE_AUTO, PREFERENCE_EMULATOR, PREFERENCE_DEVICE)


@dataclass
class Target:
    """One adb target and everything DelegateDoctor knows about it."""

    info: DeviceInfo
    manufacturer: str = ""
    avd_name: str = ""

    @property
    def serial(self) -> str:
        return self.info.serial

    @property
    def is_emulator(self) -> bool:
        return self.info.is_emulator

    @property
    def kind(self) -> str:
        return "emulator" if self.is_emulator else "physical"

    @property
    def usable(self) -> bool:
        """Can this target produce Arm64 performance evidence?"""
        return self.info.abi == REQUIRED_ABI

    @property
    def display_name(self) -> str:
        if self.avd_name:
            return self.avd_name
        if self.manufacturer and self.manufacturer.lower() not in self.info.model.lower():
            return f"{self.manufacturer} {self.info.model}"
        return self.info.model or self.serial

    @property
    def kind_label(self) -> str:
        return "Android Emulator" if self.is_emulator else "Physical Android Device"

    def describe(self) -> str:
        """Two-line form used by the interactive chooser."""
        return (f"{self.display_name}\n"
                f"   {self.kind_label}\n"
                f"   {self.info.abi} · Android {self.info.android_release}")

    def short_description(self) -> str:
        kind = "Emulator" if self.is_emulator else "Physical"
        return f"{self.display_name} · {kind} · {self.info.abi}"


def list_adb_serials() -> list:
    """Every adb target and its state, e.g. [("emulator-5554", "device")]."""
    listing = run_adb("devices")
    targets = []
    for line in listing.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and not line.startswith("*"):
            targets.append((parts[0], parts[1]))
    return targets


def _property(serial: str, name: str) -> str:
    try:
        return run_adb("shell", "getprop", name, serial=serial).strip()
    except DeviceError:
        return ""


def probe_target(serial: str) -> Target:
    """Read one target's metadata over adb."""
    info = DeviceInfo(
        serial=serial,
        model=_property(serial, "ro.product.model"),
        abi=_property(serial, "ro.product.cpu.abi"),
        android_release=_property(serial, "ro.build.version.release"),
        sdk_level=_property(serial, "ro.build.version.sdk"),
        hardware=_property(serial, "ro.hardware"),
    )
    return Target(
        info=info,
        manufacturer=_property(serial, "ro.product.manufacturer"),
        # Emulators report the AVD they were launched from; phones report "".
        avd_name=_property(serial, "ro.boot.qemu.avd_name")
                 or _property(serial, "ro.kernel.qemu.avd_name"),
    )


def discover_targets() -> list:
    """All adb targets that are actually online, with their metadata.

    Targets in `unauthorized`, `offline` or `no permissions` states are skipped:
    they cannot be measured, and probing them just produces empty properties.
    """
    targets = []
    for serial, state in list_adb_serials():
        if state != "device":
            continue
        targets.append(probe_target(serial))
    return targets


def usable_targets(targets: list) -> list:
    """Only the ones that can produce Arm64 evidence. Physical first."""
    usable = [target for target in targets if target.usable]
    # A phone outranks an emulator when both are present and nothing was asked
    # for: its numbers are the ones a user actually ships against.
    return sorted(usable, key=lambda target: target.is_emulator)


def format_target_menu(targets: list) -> str:
    lines = ["", "Available Arm targets", ""]
    for position, target in enumerate(targets, start=1):
        lines.append(f"{position}. {target.describe()}")
        lines.append("")
    return "\n".join(lines)


def _no_target_message(all_targets: list) -> str:
    if not all_targets:
        return (
            "ARM TARGET NOT FOUND\n"
            "\n"
            "No Android target is visible to adb. DelegateDoctor measures the\n"
            "model on real Arm64 hardware, so it needs one.\n"
            "\n"
            "Check with:\n"
            "\n"
            "    adb devices\n"
            "\n"
            "Then either connect a physical Arm64 Android phone with USB\n"
            "debugging enabled, or start an Arm64 emulator:\n"
            "\n"
            "    delegate-doctor setup-android\n"
        )

    listed = "\n".join(
        f"    {target.serial}  {target.display_name}  {target.info.abi or 'unknown ABI'}"
        for target in all_targets
    )
    return (
        f"ARM TARGET NOT FOUND\n"
        f"\n"
        f"{len(all_targets)} Android target(s) are attached, but none of them is\n"
        f"{REQUIRED_ABI}:\n"
        f"\n"
        f"{listed}\n"
        f"\n"
        f"DelegateDoctor targets Arm64 only, and the runners in runners/ are\n"
        f"cross-compiled for {REQUIRED_ABI}. An x86_64 emulator runs Android\n"
        f"perfectly well but says nothing about Arm performance, so it is never\n"
        f"used as a benchmark target.\n"
        f"\n"
        f"Use an {REQUIRED_ABI} system image, or a physical Arm64 phone."
    )


def select_target(
    preference: str = PREFERENCE_AUTO,
    serial: Optional[str] = None,
    interactive: bool = True,
    targets: Optional[list] = None,
    prompt=input,
    announce=print,
) -> Target:
    """Choose the one target this run will use.

    `targets` is injectable so the decision logic can be tested without adb.
    """
    if preference not in PREFERENCES:
        raise DeviceError(
            f"Unknown target preference: {preference!r}\n"
            f"Expected one of: {', '.join(PREFERENCES)}"
        )

    if targets is None:
        targets = discover_targets()

    # --- an explicit serial is an instruction, not a preference ------------
    if serial:
        for target in targets:
            if target.serial == serial:
                if not target.usable:
                    raise DeviceError(
                        f"ARM TARGET NOT FOUND\n"
                        f"\n"
                        f"{serial} is {target.info.abi or 'an unknown ABI'}, and "
                        f"DelegateDoctor requires {REQUIRED_ABI}.\n"
                        f"\n"
                        f"An x86_64 target cannot produce Arm performance evidence."
                    )
                return target
        attached = ", ".join(t.serial for t in targets) or "none"
        raise DeviceError(
            f"ARM TARGET NOT FOUND\n"
            f"\n"
            f"No attached target has serial {serial!r}.\n"
            f"\n"
            f"Attached: {attached}"
        )

    usable = usable_targets(targets)

    if preference == PREFERENCE_EMULATOR:
        usable = [target for target in usable if target.is_emulator]
        if not usable:
            # Nothing is running. If this host can give a meaningful Arm64
            # emulator and the managed AVD is already provisioned, offer to
            # start it - provisioning itself stays in setup-android.
            started = _offer_managed_emulator(interactive, announce, prompt)
            if started is not None:
                return started
            raise DeviceError(_emulator_unavailable_message())
    elif preference == PREFERENCE_DEVICE:
        usable = [target for target in usable if not target.is_emulator]
        if not usable:
            raise DeviceError(
                "ARM TARGET NOT FOUND\n"
                "\n"
                f"No physical {REQUIRED_ABI} Android device is connected.\n"
                "\n"
                "Connect a phone with USB debugging enabled and check:\n"
                "\n"
                "    adb devices\n"
            )

    if not usable:
        raise DeviceError(_no_target_message(targets))

    # --- exactly one: use it, but say which ---------------------------------
    if len(usable) == 1:
        chosen = usable[0]
        announce(f"Target                  {chosen.short_description()}")
        return chosen

    # --- several: ask, or fall back to a stated rule ------------------------
    if not interactive:
        chosen = usable[0]      # physical first, per usable_targets()
        announce(f"Target                  {chosen.short_description()} "
                 f"(first of {len(usable)}; use --device to choose)")
        return chosen

    announce(format_target_menu(usable))
    while True:
        try:
            answer = prompt(f"Select benchmark target [1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise DeviceError("No target selected.")
        if not answer:
            return usable[0]
        if answer.isdigit() and 1 <= int(answer) <= len(usable):
            return usable[int(answer) - 1]
        announce(f"Enter a number between 1 and {len(usable)}.")


def _emulator_unavailable_message() -> str:
    """Why no Arm emulator is available, tailored to this host."""
    from . import android_environment

    environment = android_environment.detect()
    support = environment.emulator_support
    if support == android_environment.SUPPORT_UNAVAILABLE:
        return (f"ARM64 EMULATOR UNAVAILABLE ON THIS HOST\n"
                f"\n"
                f"{environment.emulator_support_reason}")
    return (f"ARM TARGET NOT FOUND\n"
            f"\n"
            f"No {REQUIRED_ABI} Android emulator is running.\n"
            f"\n"
            f"Provision one with:\n"
            f"\n"
            f"    delegate-doctor setup-android\n")


def _offer_managed_emulator(interactive: bool, announce, prompt):
    """Start DelegateDoctor's own AVD, if it exists and the user agrees.

    Deliberately narrow: this starts an AVD that setup-android already created.
    It never installs packages, never creates an AVD, and never runs a long
    provisioning flow from inside `optimize`.
    """
    from . import android_environment, emulator

    environment = android_environment.detect()
    if environment.emulator_support == android_environment.SUPPORT_UNAVAILABLE:
        return None
    if not environment.can_manage_emulator:
        return None
    try:
        if not emulator.avd_exists(environment):
            return None
    except emulator.EmulatorError:
        return None

    if interactive:
        try:
            answer = prompt(
                f"Start the {emulator.AVD_NAME} emulator? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if answer not in ("", "y", "yes"):
            return None

    try:
        serial = emulator.start_delegate_doctor_emulator(environment,
                                                         announce=announce)
    except emulator.EmulatorError as error:
        raise DeviceError(str(error))

    for target in discover_targets():
        if target.serial == serial and target.usable:
            return target
    raise DeviceError(
        f"The emulator started as {serial}, but it is not a usable "
        f"{REQUIRED_ABI} target."
    )
