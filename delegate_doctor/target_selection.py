"""Find the physical Arm64 Android phones attached to this machine.

DelegateDoctor measures on a real phone, and only on a real phone. This module
discovers every adb target, keeps the physical arm64-v8a ones, and picks one -
by explicit `--device SERIAL`, automatically when there is exactly one, or by
asking when there are several.

Two things matter more than convenience here:

  * **The ABI is checked before anything is measured.** An x86_64 Android target
    tells you nothing about Arm, so it is never offered as a performance target.

  * **One serial is chosen once and carried everywhere.** Profiling, device
    verification and the benchmark all receive it explicitly, so a second phone
    appearing mid-run cannot split a before/after comparison across two devices.

Emulators are detected only so they can be excluded with a clear message: an
emulator's latency is a property of the host, and DelegateDoctor no longer
presents it as Arm evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .device import AdbNotFound, DeviceError, DeviceInfo, run_adb

REQUIRED_ABI = "arm64-v8a"


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
    def usable(self) -> bool:
        """Can this target produce Arm64 phone performance evidence?

        Physical and arm64-v8a. An emulator is excluded even when its ABI is
        right: its latency describes the host it runs on.
        """
        return self.info.abi == REQUIRED_ABI and not self.is_emulator

    @property
    def display_name(self) -> str:
        if self.avd_name:
            return self.avd_name
        if self.manufacturer and self.manufacturer.lower() not in self.info.model.lower():
            return f"{self.manufacturer} {self.info.model}"
        return self.info.model or self.serial

    @property
    def kind_label(self) -> str:
        return "Physical Android Device"

    def describe(self) -> str:
        """Two-line form used by the interactive chooser."""
        return (f"{self.display_name}\n"
                f"   {self.kind_label}\n"
                f"   {self.info.abi} · Android {self.info.android_release}")

    def short_description(self) -> str:
        return f"{self.display_name} · {self.info.abi}"


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


# --- what setup needs to know about physical devices ------------------------
#
# Setup asks a narrower question than `select_target`: is a *phone* ready, and
# if not, exactly what is wrong? "No device" and "device attached but you never
# accepted the USB debugging prompt" need different sentences, and reporting
# either as the other wastes the user's time.

PHYSICAL_READY = "READY"
PHYSICAL_NONE = "NONE"
PHYSICAL_UNAUTHORIZED = "UNAUTHORIZED"
PHYSICAL_OFFLINE = "OFFLINE"
PHYSICAL_WRONG_ABI = "WRONG_ABI"
PHYSICAL_NO_ADB = "NO_ADB"


@dataclass
class PhysicalDeviceStatus:
    """Whether a physical arm64-v8a phone is usable right now, and why not."""

    status: str
    target: Optional[Target] = None
    detail: str = ""

    @property
    def ready(self) -> bool:
        return self.status == PHYSICAL_READY


def physical_device_status() -> PhysicalDeviceStatus:
    """Classify the attached physical devices for the setup summary.

    Built on `list_adb_serials` and `probe_target` rather than a second
    discovery implementation, so setup and `optimize` always see the same
    adb, the same serials and the same properties.
    """
    try:
        serials = list_adb_serials()
    except DeviceError as error:
        # AdbNotFound is a DeviceError, and it is the one case that must not be
        # reported as "no device attached".
        if isinstance(error, AdbNotFound):
            return PhysicalDeviceStatus(PHYSICAL_NO_ADB, detail=str(error))
        return PhysicalDeviceStatus(PHYSICAL_NONE, detail=str(error))

    unauthorized = [serial for serial, state in serials if state == "unauthorized"]
    offline = [serial for serial, state in serials if state == "offline"]

    physical = []
    for serial, state in serials:
        if state != "device":
            continue
        target = probe_target(serial)
        if not target.is_emulator:
            physical.append(target)

    ready = [target for target in physical if target.usable]
    if ready:
        return PhysicalDeviceStatus(PHYSICAL_READY, target=ready[0])

    if physical:
        wrong = physical[0]
        return PhysicalDeviceStatus(
            PHYSICAL_WRONG_ABI, target=wrong,
            detail=wrong.info.abi or "unknown")

    if unauthorized:
        return PhysicalDeviceStatus(PHYSICAL_UNAUTHORIZED,
                                    detail=", ".join(unauthorized))
    if offline:
        return PhysicalDeviceStatus(PHYSICAL_OFFLINE, detail=", ".join(offline))
    return PhysicalDeviceStatus(PHYSICAL_NONE)


def usable_targets(targets: list) -> list:
    """Only the physical arm64-v8a phones."""
    return [target for target in targets if target.usable]


def format_target_menu(targets: list) -> str:
    lines = ["", "Available Arm targets", ""]
    for position, target in enumerate(targets, start=1):
        lines.append(f"{position}. {target.describe()}")
        lines.append("")
    return "\n".join(lines)


def _no_target_message(all_targets: list) -> str:
    if not all_targets:
        return (
            "NO ANDROID PHONE FOUND\n"
            "\n"
            f"DelegateDoctor measures on a physical {REQUIRED_ABI} Android\n"
            "phone, so it needs one attached.\n"
            "\n"
            "  1. Enable Developer options and USB debugging on the phone\n"
            "  2. Connect it over USB\n"
            "  3. Accept the debugging authorization prompt when it appears\n"
            "\n"
            "Then run the same command again.\n"
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
        f"cross-compiled for {REQUIRED_ABI}.\n"
        f"\n"
        f"An emulator is also never used: its latency describes the host it\n"
        f"runs on, not an Arm phone.\n"
        f"\n"
        f"Connect a physical {REQUIRED_ABI} Android phone."
    )


def select_target(
    serial: Optional[str] = None,
    interactive: bool = True,
    targets: Optional[list] = None,
    prompt=input,
    announce=print,
) -> Target:
    """Choose the one phone this run will use.

    `targets` is injectable so the decision logic can be tested without adb.
    """
    if targets is None:
        targets = discover_targets()

    # --- an explicit serial is an instruction ------------------------------
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
