"""Find the Arm64 Android targets attached to this machine, and choose one.

Policy, in one place:

    Physical arm64-v8a Android devices are the supported and validated
    benchmark target. DelegateDoctor does not provision, manage or validate
    emulator environments.

    An already-running arm64-v8a emulator is a best-effort fallback. If one is
    visible through adb and no physical phone is usable, DelegateDoctor runs
    against it rather than refusing - with a warning, and with every shareable
    artifact labelled as emulator evidence.

So an emulator is *usable* but never *preferred*: a phone always wins, and the
presence of an emulator never turns a single-phone run into a question.

Two things matter more than convenience here:

  * **The ABI is checked before anything is measured.** An x86_64 Android
    target tells you nothing about Arm, so it is never used - emulator or not.

  * **One serial is chosen once and carried everywhere.** Profiling, device
    verification and the benchmark all receive it explicitly, so a second
    target appearing mid-run cannot split a before/after comparison.
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
        """Can DelegateDoctor run against this target at all?

        The ABI, and nothing else. An arm64-v8a emulator is usable - as a
        best-effort fallback, never as validated evidence. Whether it is
        *preferred* is `usable_targets`' job, and how its results are labelled
        is the report's.
        """
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
        return ("Arm64 Android emulator" if self.is_emulator
                else "Physical Android device")

    def describe(self) -> str:
        """Two-line form used by the interactive chooser."""
        return (f"{self.display_name}\n"
                f"   {self.kind_label}\n"
                f"   {self.info.abi} · Android {self.info.android_release}")

    def short_description(self) -> str:
        kind = " (emulator)" if self.is_emulator else ""
        return f"{self.display_name} · {self.info.abi}{kind}"


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
    """Every arm64-v8a target, physical first.

    Ordering is the whole preference mechanism: a phone outranks an emulator,
    so a machine with both never has to be asked, and a non-interactive run
    picking `usable[0]` picks the phone.
    """
    usable = [target for target in targets if target.usable]
    return sorted(usable, key=lambda target: target.is_emulator)


def physical_targets(targets: list) -> list:
    return [target for target in usable_targets(targets) if not target.is_emulator]


EMULATOR_FALLBACK_NOTE = (
    "Note: emulator execution is not a validated DelegateDoctor benchmark "
    "target.\n"
    "Physical arm64-v8a Android hardware is recommended for performance "
    "results."
)


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
            "\n"
            "An already-running arm64-v8a Android emulator can also be used as\n"
            "a best-effort target. DelegateDoctor does not create or manage\n"
            "one.\n"
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
        f"An {REQUIRED_ABI} emulator would be usable as a best-effort target,\n"
        f"but an x86_64 one is not: its latency describes the host it runs on.\n"
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
                        f"UNSUPPORTED ANDROID TARGET\n"
                        f"\n"
                        f"ABI                     "
                        f"{target.info.abi or 'unknown'}\n"
                        f"Required                {REQUIRED_ABI}\n"
                        f"\n"
                        f"An x86_64 target cannot produce Arm performance "
                        f"evidence, so DelegateDoctor will not benchmark on it."
                    )
                # An explicitly requested target is never silently swapped for
                # another - including when it is an emulator. It is used, and
                # the warning says what that means.
                return _announce_choice(target, announce)
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

    physical = physical_targets(targets)

    # --- exactly one phone: use it, whatever else is running ----------------
    #
    # A running emulator must not turn a one-phone machine into a question.
    if len(physical) == 1:
        return _announce_choice(physical[0], announce)

    # --- no phone at all: fall back to an emulator, best-effort -------------
    if not physical:
        if len(usable) == 1:
            return _announce_choice(usable[0], announce)
        return _choose_among(usable, interactive, prompt, announce,
                             "emulator")

    # --- several phones: ask, or say to use --device ------------------------
    return _choose_among(physical, interactive, prompt, announce, "phone")


def _announce_choice(chosen: "Target", announce) -> "Target":
    """Name the target, and warn if it is a best-effort emulator."""
    announce(f"Android target          {chosen.display_name}")
    announce(f"Type                    {chosen.kind_label}")
    announce(f"ABI                     {chosen.info.abi}")
    if chosen.is_emulator:
        announce("")
        announce(EMULATOR_FALLBACK_NOTE)
    return chosen


def _choose_among(candidates: list, interactive: bool, prompt, announce,
                  noun: str) -> "Target":
    """Several equally-ranked targets: ask, or explain how to disambiguate."""
    if not interactive:
        chosen = candidates[0]
        announce(f"Android target          {chosen.display_name} "
                 f"(first of {len(candidates)}; use --device to choose)")
        if chosen.is_emulator:
            announce("")
            announce(EMULATOR_FALLBACK_NOTE)
        return chosen

    announce(format_target_menu(candidates))
    while True:
        try:
            answer = prompt(f"Select benchmark {noun} [1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise DeviceError("No target selected.")
        if not answer:
            return _announce_choice(candidates[0], announce)
        if answer.isdigit() and 1 <= int(answer) <= len(candidates):
            return _announce_choice(candidates[int(answer) - 1], announce)
        announce(f"Enter a number between 1 and {len(candidates)}.")
