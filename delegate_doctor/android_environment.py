"""What Android tooling this machine actually has, as structured data.

Everything here answers questions - it changes nothing. `emulator.py` and
`android_setup.py` do the provisioning; this module just looks, so the looking
can be unit-tested without an Android SDK anywhere in sight.

The rule that shapes the whole file: **discover, never search.** Conventional
environment variables and a short list of standard install locations are
checked, and that is all. DelegateDoctor does not walk a user's filesystem
hunting for an SDK, and does not read anything outside the SDK it finds.

Host support is deliberately three-valued, because "an Arm emulator could
theoretically run here" and "DelegateDoctor has measured on it" are different
claims and only one of them belongs in a performance tool.
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --- host ------------------------------------------------------------------

OS_MACOS = "macOS"
OS_WINDOWS = "Windows"
OS_LINUX = "Linux"
OS_UNKNOWN = "unknown"

ARCH_ARM64 = "arm64"
ARCH_X86_64 = "x86_64"
ARCH_UNKNOWN = "unknown"

# How the same machine spells itself on different platforms.
_ARCH_ALIASES = {
    "arm64": ARCH_ARM64, "aarch64": ARCH_ARM64, "armv8": ARCH_ARM64,
    "arm64e": ARCH_ARM64,
    "x86_64": ARCH_X86_64, "amd64": ARCH_X86_64, "x64": ARCH_X86_64,
    "i386": ARCH_X86_64, "i686": ARCH_X86_64,
}

_OS_ALIASES = {"darwin": OS_MACOS, "windows": OS_WINDOWS, "linux": OS_LINUX}

# Emulator support, from strongest claim to weakest.
SUPPORT_VALIDATED = "SUPPORTED"        # DelegateDoctor has measured on this
SUPPORT_UNTESTED = "UNTESTED"          # plausible, never validated here
SUPPORT_UNAVAILABLE = "UNAVAILABLE"    # cannot give a meaningful Arm emulator


@dataclass(frozen=True)
class HostPlatform:
    """The machine DelegateDoctor is running on, normalized."""

    os_name: str
    architecture: str
    raw_system: str = ""
    raw_machine: str = ""

    @property
    def is_windows(self) -> bool:
        return self.os_name == OS_WINDOWS

    @property
    def is_macos(self) -> bool:
        return self.os_name == OS_MACOS

    @property
    def is_arm64(self) -> bool:
        return self.architecture == ARCH_ARM64

    def describe(self) -> str:
        return f"{self.os_name} {self.architecture}"


def normalize_architecture(machine: str) -> str:
    return _ARCH_ALIASES.get((machine or "").strip().lower(), ARCH_UNKNOWN)


def normalize_os(system: str) -> str:
    return _OS_ALIASES.get((system or "").strip().lower(), OS_UNKNOWN)


def detect_host(system: str = None, machine: str = None) -> HostPlatform:
    """The current host. Both inputs are injectable so every case is testable."""
    raw_system = platform.system() if system is None else system
    raw_machine = platform.machine() if machine is None else machine
    return HostPlatform(
        os_name=normalize_os(raw_system),
        architecture=normalize_architecture(raw_machine),
        raw_system=raw_system,
        raw_machine=raw_machine,
    )


def emulator_support(host: HostPlatform) -> tuple:
    """Can this host give a *meaningful Arm64* Android emulator? (status, reason)

    An Android emulator only runs Arm64 code at usable speed when the host is
    itself Arm64 - otherwise it is emulating one instruction set on another,
    and any latency it reports is a property of the emulation, not of Arm.

    So an x86_64 host is told UNAVAILABLE rather than being handed an x86_64
    image: DelegateDoctor measures Arm, and an x86 number presented as an Arm
    number would be worse than no number at all.
    """
    if host.os_name == OS_MACOS and host.architecture == ARCH_ARM64:
        return SUPPORT_VALIDATED, (
            "Apple Silicon runs arm64-v8a system images natively. This is the "
            "configuration DelegateDoctor's emulator evidence was measured on."
        )

    if host.architecture == ARCH_ARM64 and host.os_name in (OS_LINUX, OS_WINDOWS):
        return SUPPORT_UNTESTED, (
            f"{host.os_name} on arm64 could in principle run an arm64-v8a "
            f"system image natively, but DelegateDoctor has never been "
            f"validated on it. Treat any emulator result from this host as "
            f"unverified, and prefer a physical device."
        )

    if host.architecture == ARCH_X86_64:
        return SUPPORT_UNAVAILABLE, (
            f"This host is {host.architecture}. An arm64-v8a system image "
            f"would have to be emulated instruction by instruction, and its "
            f"latency would measure the emulation rather than Arm.\n"
            f"\n"
            f"DelegateDoctor will not provision an x86_64 emulator and call it "
            f"an Arm target. Connect a physical arm64-v8a Android device."
        )

    return SUPPORT_UNAVAILABLE, (
        f"Unrecognised host ({host.raw_system} {host.raw_machine}). "
        f"Connect a physical arm64-v8a Android device."
    )


# --- SDK and tools ----------------------------------------------------------

SDK_ENVIRONMENT_VARIABLES = ("ANDROID_HOME", "ANDROID_SDK_ROOT")
NDK_ENVIRONMENT_VARIABLES = ("ANDROID_NDK_HOME", "ANDROID_NDK_ROOT", "ANDROID_NDK")

# Standard install locations only. This list is short on purpose: a longer one
# would be a filesystem search wearing a disguise.
DEFAULT_SDK_LOCATIONS = (
    "~/Library/Android/sdk",                          # Android Studio, macOS
    "~/Android/Sdk",                                  # Android Studio, Linux
    "~/AppData/Local/Android/Sdk",                    # Android Studio, Windows
    "/opt/homebrew/share/android-commandlinetools",   # Homebrew cask
    "/usr/local/share/android-commandlinetools",
)

# Tool name -> (subdirectory inside the SDK, POSIX name, Windows name).
# sdkmanager and avdmanager ship as .bat files on Windows, not .exe.
TOOL_LAYOUT = {
    "adb": ("platform-tools", "adb", "adb.exe"),
    "emulator": ("emulator", "emulator", "emulator.exe"),
    "sdkmanager": ("cmdline-tools", "sdkmanager", "sdkmanager.bat"),
    "avdmanager": ("cmdline-tools", "avdmanager", "avdmanager.bat"),
}


@dataclass(frozen=True)
class AndroidTool:
    """One external tool, and where it was found."""

    name: str
    path: Optional[Path] = None

    @property
    def found(self) -> bool:
        return self.path is not None

    @property
    def status(self) -> str:
        return "FOUND" if self.found else "MISSING"


def _executable_name(tool: str, host: HostPlatform) -> str:
    _, posix_name, windows_name = TOOL_LAYOUT[tool]
    return windows_name if host.is_windows else posix_name


def is_usable_sdk(path: Path) -> bool:
    """Does this directory actually contain Android tooling?

    A directory existing proves nothing: `~/Library/Android/sdk` survives an
    uninstall, and an interrupted download leaves a folder behind. So the test
    is whether one of the tools DelegateDoctor needs is really there.
    """
    path = Path(path)
    if not path.is_dir():
        return False
    if (path / "platform-tools").is_dir() or (path / "cmdline-tools").is_dir():
        return True
    return (path / "emulator").is_dir()


def find_sdk_root(environment: dict = None, home: Path = None) -> Optional[Path]:
    """The Android SDK Android Studio installed, in one documented order.

        1. ANDROID_HOME
        2. ANDROID_SDK_ROOT
        3. the standard Android Studio install location for this platform

    There is no fourth entry. DelegateDoctor used to fall back to bootstrapping
    a private SDK from Google's command-line tools archive; that is gone.
    Android Studio's Setup Wizard is the one manual Android prerequisite, and
    maintaining a second way to obtain an SDK meant two onboarding paths, two
    sets of failure modes, and a checksum table nobody could verify offline.
    """
    source = os.environ if environment is None else environment

    for variable in SDK_ENVIRONMENT_VARIABLES:
        value = source.get(variable)
        if value:
            candidate = Path(value).expanduser()
            # An explicitly-set variable is honoured even if it looks empty:
            # the user said where it is, and second-guessing that would be
            # worse than reporting the missing tools a moment later.
            if candidate.is_dir():
                return candidate

    for location in DEFAULT_SDK_LOCATIONS:
        candidate = Path(location).expanduser()
        if is_usable_sdk(candidate):
            return candidate

    return None


def _cmdline_tools_binary(sdk_root: Path, name: str) -> Optional[Path]:
    """`cmdline-tools/<version>/bin/<name>`, preferring `latest`.

    Google versions this directory, so the exact path is not fixed. Only the
    direct children of cmdline-tools are examined - no deep walk.
    """
    parent = sdk_root / "cmdline-tools"
    if not parent.is_dir():
        return None

    versions = []
    latest = parent / "latest"
    if latest.is_dir():
        versions.append(latest)
    try:
        versions += sorted((child for child in parent.iterdir()
                            if child.is_dir() and child.name != "latest"),
                           reverse=True)
    except OSError:
        return None

    for version_dir in versions:
        candidate = version_dir / "bin" / name
        if candidate.is_file():
            return candidate
    return None


def find_tool(tool: str, host: HostPlatform, sdk_root: Optional[Path],
              path_lookup=shutil.which) -> AndroidTool:
    """Locate one tool inside the SDK, falling back to PATH.

    The SDK copy is preferred: a tool found there is guaranteed to match the
    SDK being managed, whereas a PATH hit could be from an unrelated install.
    """
    executable = _executable_name(tool, host)

    if sdk_root is not None:
        subdirectory = TOOL_LAYOUT[tool][0]
        if subdirectory == "cmdline-tools":
            found = _cmdline_tools_binary(sdk_root, executable)
            if found is not None:
                return AndroidTool(tool, found)
        else:
            candidate = sdk_root / subdirectory / executable
            if candidate.is_file():
                return AndroidTool(tool, candidate)

    on_path = path_lookup(executable) or path_lookup(tool)
    if on_path:
        return AndroidTool(tool, Path(on_path))
    return AndroidTool(tool, None)


def is_ndk_directory(path: Path) -> bool:
    """An NDK is identified by the CMake toolchain file the runner build needs."""
    return (path / "build" / "cmake" / "android.toolchain.cmake").is_file()


def newest_ndk_in_sdk(sdk_root: Path) -> Optional[Path]:
    """The highest-numbered NDK inside an SDK, if any."""
    ndk_parent = sdk_root / "ndk"
    if not ndk_parent.is_dir():
        return None
    try:
        candidates = [child for child in ndk_parent.iterdir()
                      if is_ndk_directory(child)]
    except OSError:
        return None
    if not candidates:
        return None

    def version_key(path: Path):
        return [int(piece) if piece.isdigit() else 0
                for piece in path.name.split(".")]

    return sorted(candidates, key=version_key)[-1]


def find_ndk(sdk_root: Optional[Path] = None,
             environment: dict = None) -> Optional[Path]:
    """An Android NDK, preferring an explicit environment variable."""
    source = os.environ if environment is None else environment

    for variable in NDK_ENVIRONMENT_VARIABLES:
        value = source.get(variable)
        if value:
            candidate = Path(value).expanduser()
            if is_ndk_directory(candidate):
                return candidate

    roots = []
    for variable in SDK_ENVIRONMENT_VARIABLES:
        value = source.get(variable)
        if value:
            roots.append(Path(value).expanduser())
    if sdk_root is not None:
        roots.append(sdk_root)
    for location in DEFAULT_SDK_LOCATIONS:
        roots.append(Path(location).expanduser())

    for root in roots:
        if not root.is_dir():
            continue
        found = newest_ndk_in_sdk(root)
        if found is not None:
            return found
    return None


@dataclass
class AndroidEnvironment:
    """Everything DelegateDoctor could determine about this machine's Android setup."""

    host: HostPlatform
    sdk_root: Optional[Path] = None
    tools: dict = field(default_factory=dict)
    ndk: Optional[Path] = None
    cmake: Optional[Path] = None
    git: Optional[Path] = None

    def tool(self, name: str) -> AndroidTool:
        return self.tools.get(name, AndroidTool(name, None))

    def tool_path(self, name: str) -> Optional[Path]:
        return self.tool(name).path

    @property
    def has_sdk(self) -> bool:
        return self.sdk_root is not None

    @property
    def has_command_line_tools(self) -> bool:
        return self.tool("sdkmanager").found and self.tool("avdmanager").found

    @property
    def can_manage_emulator(self) -> bool:
        """Enough tooling present to create and start an AVD."""
        return (self.has_command_line_tools
                and self.tool("emulator").found
                and self.tool("adb").found)

    @property
    def emulator_support(self) -> str:
        return emulator_support(self.host)[0]

    @property
    def emulator_support_reason(self) -> str:
        return emulator_support(self.host)[1]


def detect(environment: dict = None, host: HostPlatform = None,
           path_lookup=shutil.which, home: Path = None) -> AndroidEnvironment:
    """Inspect the machine once, and hand back a value object."""
    source = os.environ if environment is None else environment
    host = host or detect_host()
    sdk_root = find_sdk_root(source, home=home)

    tools = {name: find_tool(name, host, sdk_root, path_lookup)
             for name in TOOL_LAYOUT}

    return AndroidEnvironment(
        host=host,
        sdk_root=sdk_root,
        tools=tools,
        ndk=find_ndk(sdk_root, source),
        cmake=_optional_path(path_lookup("cmake")),
        git=_optional_path(path_lookup("git")),
    )


def _optional_path(value) -> Optional[Path]:
    return Path(value) if value else None


COMMAND_LINE_TOOLS_MISSING_MESSAGE = (
    "Android SDK found, but Android command-line tools are missing.\n"
    "\n"
    "DelegateDoctor uses Google's own sdkmanager and avdmanager to inspect and\n"
    "install Android packages. It does not download or bundle them itself.\n"
    "\n"
    "Install the Android SDK Command-line Tools - through Android Studio's SDK\n"
    "Manager, or by unpacking Google's commandlinetools archive into:\n"
    "\n"
    "    <SDK>/cmdline-tools/latest/\n"
    "\n"
    "then run setup again."
)

SDK_MISSING_MESSAGE = (
    "Android SDK             NOT FOUND\n"
    "\n"
    "Install Android Studio and complete the initial Setup Wizard.\n"
    "Then rerun:\n"
    "\n"
    "    delegate-doctor setup-android\n"
    "\n"
    "DelegateDoctor looked at ANDROID_HOME, ANDROID_SDK_ROOT and the standard\n"
    "Android Studio location. It does not search the filesystem, and it does\n"
    "not download an SDK of its own.\n"
    "\n"
    "If your SDK is somewhere unusual, point at it:\n"
    "\n"
    "    export ANDROID_HOME=/path/to/Android/sdk\n"
    "\n"
    "The SDK is needed either way: a physical Arm64 phone with adb also works\n"
    "as a target, and adb ships with the SDK's platform-tools."
)
