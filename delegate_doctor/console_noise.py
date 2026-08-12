"""Hide a short allowlist of upstream messages that are benign *here*.

A successful DelegateDoctor run used to print a dozen PyTorch and ExecuTorch
diagnostics that have nothing to do with the result, which made a working run
look broken. This module removes exactly those, and nothing else.

The rule, which every function below obeys:

    a message on the allowlist   ->  hidden in normal mode
    anything else                ->  still shown, unchanged

There is no `warnings.filterwarnings("ignore")` here, no blanket stderr
redirection, and no logger disabled wholesale. An unknown warning, an unexpected
native error or a DelegateDoctor diagnostic always reaches the user.

Four mechanisms, because the noise arrives four different ways
---------------------------------------------------------------
Each was identified by capturing it in the pinned environment (ExecuTorch 1.4.0
/ torch 2.13.0) rather than guessed at:

  1. `warnings.warn`      LeafSpec FutureWarning, Inspector UserWarning
                          -> scoped `warnings.catch_warnings()`
  2. `logging`            the two torch messages printed during *import*
                          -> a message-matching filter on two named loggers
  3. native stderr (fd 2) ExecuTorch's C++ cpuinfo probe
                          -> capture fd 2, replay every unrecognised line
  4. nothing else.

Patterns match message *content* and category, never absolute file paths, PIDs,
timestamps or C++ line numbers, all of which move between builds and machines.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import sys
import tempfile
import warnings

# --- 1. known Python warnings ----------------------------------------------
#
# (category, compiled message pattern, why it is safe to hide)

KNOWN_WARNINGS = (
    (
        FutureWarning,
        re.compile(r"isinstance\(treespec, LeafSpec\).*deprecated", re.DOTALL),
        # torch.export's own pytree code trips this on every deepcopy of an
        # ExportedProgram. DelegateDoctor deep-copies the graph several times
        # per run, so one upstream deprecation becomes dozens of lines. It says
        # nothing about the model, the repair or the measurement.
        "torch.export internal pytree deprecation, emitted once per deepcopy",
    ),
    (
        UserWarning,
        re.compile(r"Output Buffer not found\. Tensor Debug Data will not be "
                   r"available"),
        # ExecuTorch's Inspector says this when an ETDump carries no debug
        # buffer. DelegateDoctor deliberately profiles without one: it reads
        # event durations, and reads tensors back through a separate untimed
        # device-verification run instead.
        "ETDump has no debug buffer; DelegateDoctor does not read tensors from it",
    ),
)


# --- 2. known logging records, emitted while torch/executorch import --------
#
# (logger name, message substring, why it is safe to hide)

KNOWN_LOG_RECORDS = (
    (
        "torch.distributed.elastic.multiprocessing.redirects",
        "Redirects are currently not supported in MacOs",
        # torch.distributed announces a macOS limitation of its subprocess
        # redirect helper. DelegateDoctor never uses torch.distributed.
        "torch.distributed is not part of any DelegateDoctor code path",
    ),
    (
        "torch.utils._pytree",
        "register_constant() on Enum subclasses is deprecated",
        # An upstream deprecation about how ExecuTorch registers one of its own
        # enums with pytree. It concerns ExecuTorch's future compatibility, not
        # this run.
        "upstream deprecation in how ExecuTorch registers its own enum",
    ),
)


# --- 3. known native stderr lines -------------------------------------------
#
# ExecuTorch's C++ runtime probes CPU information on startup and narrates it.
# The paths it reports are Android ones that do not exist on a developer's
# machine, so "Failed to open" is the expected result of a probe, not an error.
# Matched without the `:NN` source line, which moves between builds.

KNOWN_NATIVE_STDERR = (
    re.compile(r"^\[cpuinfo_utils\.cpp:\d+\]\s+Reading file /sys/devices/soc0/"),
    re.compile(r"^\[cpuinfo_utils\.cpp:\d+\]\s+Failed to open midr file"),
    re.compile(r"^\[cpuinfo_utils\.cpp:\d+\]\s+Number of efficient cores"),
)

# Set by `quieten_import_logging()`, so it can be undone.
_INSTALLED_LOG_FILTERS: list = []

# Escape hatch: set this to see the import-time messages the filters below hide.
VERBOSE_IMPORT_ENVIRONMENT_VARIABLE = "DELEGATE_DOCTOR_VERBOSE_IMPORTS"


def is_known_warning(category, message: str) -> bool:
    """Is this one of the upstream warnings on the allowlist?"""
    text = str(message)
    return any(issubclass(category, known) and pattern.search(text)
               for known, pattern, _ in KNOWN_WARNINGS)


def is_known_native_line(line: str) -> bool:
    """Is this native stderr line one of the benign ExecuTorch probe lines?"""
    stripped = line.strip()
    return any(pattern.search(stripped) for pattern in KNOWN_NATIVE_STDERR)


# --- scoped suppression -----------------------------------------------------


@contextlib.contextmanager
def suppress_known_warnings(verbose: bool = False):
    """Hide the allowlisted warnings for the duration of the block.

    Uses `warnings.catch_warnings()`, so the caller's filter configuration is
    saved on entry and restored on exit - including when the block raises.
    DelegateDoctor is a library as well as a tool, and importing it must not
    change how warnings behave anywhere else in the user's program.

    Only "ignore" entries for the allowlisted patterns are added. Every other
    warning keeps whatever behaviour the caller already configured.
    """
    if verbose:
        yield
        return

    with warnings.catch_warnings():
        for category, pattern, _ in KNOWN_WARNINGS:
            # `filterwarnings` anchors its regex with `re.match`, while the
            # patterns above are written for `search`. The real LeafSpec text
            # begins with a backtick, so without this leading `.*` the filter
            # would silently never match.
            warnings.filterwarnings("ignore", message=f".*{pattern.pattern}",
                                    category=category)
        yield


@contextlib.contextmanager
def filter_native_stderr(verbose: bool = False, suppressed: list = None):
    """Drop benign native ExecuTorch probe lines; replay everything else.

    ExecuTorch's C++ runtime writes straight to file descriptor 2, where
    Python's `warnings` and `logging` machinery cannot reach it. So fd 2 is
    briefly pointed at a temporary file, and afterwards every captured line is
    examined: recognised probe lines are dropped, and **everything else is
    written back to the real stderr in order**.

    That last part is the point. A genuine runtime failure printed by native
    code still reaches the user; it is never swallowed.

    `suppressed` optionally collects the dropped lines, so they can be written
    to the run's `runtime.log`.
    """
    if verbose:
        yield
        return

    try:
        stderr_fileno = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        # Under pytest's capture, or any wrapper without a real descriptor,
        # there is nothing at fd level to capture. Do nothing rather than
        # guess: a missing filter is far better than lost output.
        yield
        return

    saved = os.dup(stderr_fileno)
    buffer = tempfile.TemporaryFile(mode="w+b")
    try:
        sys.stderr.flush()
        os.dup2(buffer.fileno(), stderr_fileno)
        try:
            yield
        finally:
            # Restore first, so the replay below cannot land back in the buffer,
            # and so an exception leaves stderr working.
            sys.stderr.flush()
            os.dup2(saved, stderr_fileno)
    finally:
        os.close(saved)
        try:
            buffer.seek(0)
            captured = buffer.read().decode("utf-8", errors="replace")
        finally:
            buffer.close()
        replay_native_stderr(captured, suppressed)


def replay_native_stderr(captured: str, suppressed: list = None) -> str:
    """Write back every line that is not on the allowlist. Returns what was kept."""
    kept = []
    for line in captured.splitlines():
        if is_known_native_line(line):
            if suppressed is not None:
                suppressed.append(line)
        else:
            kept.append(line)

    text = "\n".join(kept)
    if text.strip():
        print(text, file=sys.stderr, flush=True)
    return text


@contextlib.contextmanager
def suppress_known_noise(verbose: bool = False, suppressed: list = None):
    """Both scoped filters at once, for a block that can emit either."""
    with suppress_known_warnings(verbose=verbose):
        with filter_native_stderr(verbose=verbose, suppressed=suppressed):
            yield


# --- import-time logging ----------------------------------------------------


class KnownMessageFilter(logging.Filter):
    """Drops one specific known message; passes everything else through.

    Deliberately not a level change: the logger keeps working normally, and
    every other record it emits - including real warnings and errors from the
    same module - is untouched.
    """

    def __init__(self, needle: str):
        super().__init__()
        self.needle = needle

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return self.needle not in record.getMessage()
        except Exception:
            # An unformattable record is not ours to judge; let it through.
            return True


def quieten_import_logging() -> None:
    """Filter the two torch log records emitted while ExecuTorch imports.

    This is the one thing here that happens at import time, and it is the only
    option available: both records are emitted *during* `import executorch`, so
    a context manager around a later operation would run far too late.

    The footprint is deliberately tiny - two named loggers, two exact message
    substrings, no root logger, no level changes, and nothing at all touched in
    the `warnings` module. `restore_import_logging()` undoes it, and setting
    DELEGATE_DOCTOR_VERBOSE_IMPORTS=1 skips it entirely.
    """
    if os.environ.get(VERBOSE_IMPORT_ENVIRONMENT_VARIABLE):
        return
    if _INSTALLED_LOG_FILTERS:
        return

    for logger_name, needle, _ in KNOWN_LOG_RECORDS:
        logger = logging.getLogger(logger_name)
        log_filter = KnownMessageFilter(needle)
        logger.addFilter(log_filter)
        _INSTALLED_LOG_FILTERS.append((logger, log_filter))


def restore_import_logging() -> None:
    """Remove the filters installed by `quieten_import_logging()`."""
    while _INSTALLED_LOG_FILTERS:
        logger, log_filter = _INSTALLED_LOG_FILTERS.pop()
        logger.removeFilter(log_filter)


def describe_policy() -> str:
    """The allowlist in words, for `runtime.log` and for anyone auditing it."""
    lines = ["DelegateDoctor suppressed these known-benign upstream messages.",
             "Everything not listed here was passed through untouched.",
             "", "Python warnings:"]
    for category, pattern, why in KNOWN_WARNINGS:
        lines.append(f"  {category.__name__}: {pattern.pattern}")
        lines.append(f"    reason: {why}")
    lines.append("")
    lines.append("Log records (filtered at import):")
    for logger_name, needle, why in KNOWN_LOG_RECORDS:
        lines.append(f"  {logger_name}: {needle}")
        lines.append(f"    reason: {why}")
    lines.append("")
    lines.append("Native stderr lines:")
    for pattern in KNOWN_NATIVE_STDERR:
        lines.append(f"  {pattern.pattern}")
    return "\n".join(lines)
