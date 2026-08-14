"""Only the allowlist is hidden. Everything else survives.

The dangerous failure mode for this module is not "a warning slipped through" -
it is "a real problem was silently swallowed". Most of what follows tests the
second thing.

Offline: the exact upstream message strings from the pinned environment are
used as literals, so nothing here has to run PyTorch or ExecuTorch to be
meaningful.
"""

import logging
import subprocess
import sys
import warnings

import pytest

from delegate_doctor import console_noise

# --- the exact messages, as captured from ExecuTorch 1.4.0 / torch 2.13.0 ---

LEAFSPEC = ("`isinstance(treespec, LeafSpec)` is deprecated, use "
            "`isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.")

INSPECTOR = "Output Buffer not found. Tensor Debug Data will not be available."

REDIRECTS = "NOTE: Redirects are currently not supported in MacOs."

KERNEL_PREFERENCE = (
    "<enum 'KernelPreference'> is an Enum subclass and is now natively "
    "supported by torch.compile as an opaque value type. Calling "
    "register_constant() on Enum subclasses is deprecated and will be an "
    "error in a future release."
)

CPUINFO_LINES = [
    "[cpuinfo_utils.cpp:71] Reading file /sys/devices/soc0/image_version",
    "[cpuinfo_utils.cpp:87] Failed to open midr file /sys/devices/soc0/image_version",
    "[cpuinfo_utils.cpp:167] Number of efficient cores 4",
]


# --- recognition -------------------------------------------------------------

def test_the_leafspec_warning_is_recognised():
    assert console_noise.is_known_warning(FutureWarning, LEAFSPEC)


def test_the_inspector_warning_is_recognised():
    assert console_noise.is_known_warning(UserWarning, INSPECTOR)


@pytest.mark.parametrize("line", CPUINFO_LINES)
def test_the_cpuinfo_lines_are_recognised(line):
    assert console_noise.is_known_native_line(line)


@pytest.mark.parametrize("line_number", [1, 42, 999, 100000])
def test_cpuinfo_matching_does_not_depend_on_source_line_numbers(line_number):
    """C++ line numbers move between builds; the message content does not."""
    line = f"[cpuinfo_utils.cpp:{line_number}] Number of efficient cores 8"
    assert console_noise.is_known_native_line(line)


def test_the_category_must_match_too():
    """The same text under a different category is not on the allowlist."""
    assert not console_noise.is_known_warning(UserWarning, LEAFSPEC)
    assert not console_noise.is_known_warning(FutureWarning, INSPECTOR)


@pytest.mark.parametrize("text", [
    "something else entirely",
    "isinstance(treespec, TreeSpec) is fine",
    "Output Buffer found",
    "",
])
def test_unrelated_text_is_not_recognised(text):
    assert not console_noise.is_known_warning(FutureWarning, text)
    assert not console_noise.is_known_warning(UserWarning, text)


@pytest.mark.parametrize("line", [
    "[cpuinfo_utils.cpp:87] Something completely different",
    "[other_file.cpp:87] Failed to open midr file",
    "FATAL: failed to load method",
    "[cpuinfo_utils.cpp] no line number at all",
])
def test_unrelated_native_lines_are_not_recognised(line):
    assert not console_noise.is_known_native_line(line)


def test_no_absolute_paths_appear_in_any_pattern():
    """Patterns must travel between machines and platforms."""
    patterns = [pattern.pattern for _, pattern, _ in console_noise.KNOWN_WARNINGS]
    patterns += [pattern.pattern for pattern in console_noise.KNOWN_NATIVE_STDERR]
    for pattern in patterns:
        for fragment in ("/opt/", "/usr/", ".venv", "site-packages", "anaconda",
                         "C:\\", "/home/", "/Users/"):
            assert fragment not in pattern, f"{pattern} hard-codes a local path"


# --- scoped warning suppression ---------------------------------------------

def test_a_known_warning_is_hidden_in_normal_mode(recwarn):
    with console_noise.suppress_known_warnings(verbose=False):
        warnings.warn(LEAFSPEC, FutureWarning)
        warnings.warn(INSPECTOR, UserWarning)
    assert len(recwarn) == 0


def test_a_known_warning_is_visible_in_verbose_mode():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with console_noise.suppress_known_warnings(verbose=True):
            warnings.warn(LEAFSPEC, FutureWarning)
    assert len(caught) == 1
    assert "LeafSpec" in str(caught[0].message)


def test_an_unknown_warning_still_reaches_the_user():
    """The critical case: suppression must be an allowlist, not a mute button."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with console_noise.suppress_known_warnings(verbose=False):
            warnings.warn("DelegateDoctor test warning that must remain visible",
                          UserWarning)
    assert len(caught) == 1
    assert "must remain visible" in str(caught[0].message)


def test_other_future_warnings_are_not_swept_up():
    """Only the one FutureWarning is hidden, not the category."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with console_noise.suppress_known_warnings(verbose=False):
            warnings.warn("an unrelated future change", FutureWarning)
    assert len(caught) == 1


def test_other_user_warnings_are_not_swept_up():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with console_noise.suppress_known_warnings(verbose=False):
            warnings.warn("some other ExecuTorch Inspector warning", UserWarning)
    assert len(caught) == 1


# --- the caller's global state is restored ----------------------------------

def test_the_global_warning_filters_are_restored():
    """Importing a library must not change warnings everywhere else."""
    before = list(warnings.filters)
    with console_noise.suppress_known_warnings(verbose=False):
        warnings.warn(LEAFSPEC, FutureWarning)
    assert warnings.filters == before


def test_the_filters_are_restored_even_when_the_block_raises():
    before = list(warnings.filters)
    with pytest.raises(RuntimeError):
        with console_noise.suppress_known_warnings(verbose=False):
            raise RuntimeError("boom")
    assert warnings.filters == before


def test_a_user_filter_set_inside_does_not_leak_out():
    before = list(warnings.filters)
    with console_noise.suppress_known_warnings(verbose=False):
        warnings.filterwarnings("ignore", message="something the user chose")
    assert warnings.filters == before


def test_importing_delegate_doctor_installs_no_warning_filters():
    """The package touches `logging` at import, never `warnings`."""
    import importlib

    before = list(warnings.filters)
    importlib.reload(importlib.import_module("delegate_doctor"))
    assert warnings.filters == before


# --- exceptions propagate ----------------------------------------------------

def test_exceptions_propagate_through_the_warning_filter():
    with pytest.raises(RuntimeError, match="real failure"):
        with console_noise.suppress_known_warnings():
            raise RuntimeError("real failure")


def test_exceptions_propagate_through_the_stderr_filter():
    with pytest.raises(ValueError, match="real failure"):
        with console_noise.filter_native_stderr():
            raise ValueError("real failure")


def test_exceptions_propagate_through_the_combined_filter():
    with pytest.raises(KeyError):
        with console_noise.suppress_known_noise():
            raise KeyError("real failure")


def test_a_subprocess_failure_is_not_swallowed():
    with pytest.raises(subprocess.CalledProcessError):
        with console_noise.suppress_known_noise():
            subprocess.run([sys.executable, "-c", "raise SystemExit(3)"], check=True)


# --- native stderr replay ----------------------------------------------------

def test_known_native_lines_are_dropped(capsys):
    console_noise.replay_native_stderr("\n".join(CPUINFO_LINES))
    assert capsys.readouterr().err == ""


def test_an_unknown_native_line_is_replayed(capsys):
    """The acceptance criterion: a real error between benign lines survives."""
    captured = "\n".join([
        CPUINFO_LINES[1],
        "REAL EXECUTORCH ERROR: kernel crashed",
        CPUINFO_LINES[2],
    ])
    console_noise.replay_native_stderr(captured)
    err = capsys.readouterr().err
    assert "REAL EXECUTORCH ERROR: kernel crashed" in err
    assert "cpuinfo_utils.cpp" not in err


def test_replayed_lines_keep_their_order(capsys):
    console_noise.replay_native_stderr(
        "first problem\n" + CPUINFO_LINES[0] + "\nsecond problem")
    err = capsys.readouterr().err
    assert err.index("first problem") < err.index("second problem")


def test_suppressed_native_lines_can_be_collected():
    suppressed = []
    console_noise.replay_native_stderr(
        "\n".join(CPUINFO_LINES + ["a real error"]), suppressed)
    assert len(suppressed) == 3
    assert all("cpuinfo" in line for line in suppressed)


def test_nothing_is_printed_when_everything_was_benign(capsys):
    console_noise.replay_native_stderr("\n".join(CPUINFO_LINES))
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_the_stderr_filter_captures_and_replays_at_fd_level(tmp_path):
    """End to end through a real file descriptor, using a child process.

    A subprocess is the only honest way to test fd-level capture: pytest has
    already replaced this process's stderr.
    """
    script = tmp_path / "emit.py"
    script.write_text(
        "import sys\n"
        "from delegate_doctor import console_noise\n"
        "suppressed = []\n"
        "with console_noise.filter_native_stderr(suppressed=suppressed):\n"
        "    sys.stderr.write('[cpuinfo_utils.cpp:87] Failed to open midr file x\\n')\n"
        "    sys.stderr.write('REAL EXECUTORCH ERROR: kernel crashed\\n')\n"
        "    sys.stderr.write('[cpuinfo_utils.cpp:167] Number of efficient cores 4\\n')\n"
        "    sys.stderr.flush()\n"
        "print('suppressed=%d' % len(suppressed))\n"
    )
    completed = subprocess.run([sys.executable, str(script)],
                               capture_output=True, text=True, check=True)
    assert "REAL EXECUTORCH ERROR: kernel crashed" in completed.stderr
    assert "cpuinfo_utils.cpp" not in completed.stderr
    assert "suppressed=2" in completed.stdout


def test_the_stderr_filter_passes_everything_through_in_verbose_mode(tmp_path):
    script = tmp_path / "emit_verbose.py"
    script.write_text(
        "import sys\n"
        "from delegate_doctor import console_noise\n"
        "with console_noise.filter_native_stderr(verbose=True):\n"
        "    sys.stderr.write('[cpuinfo_utils.cpp:167] Number of efficient cores 4\\n')\n"
        "    sys.stderr.flush()\n"
    )
    completed = subprocess.run([sys.executable, str(script)],
                               capture_output=True, text=True, check=True)
    assert "cpuinfo_utils.cpp" in completed.stderr


def test_the_stderr_filter_does_nothing_when_there_is_no_real_descriptor():
    """Under pytest capture there is no fd to redirect; that must be harmless."""
    with console_noise.filter_native_stderr():
        pass


# --- import-time logging filters ---------------------------------------------

def test_the_known_log_records_are_dropped(caplog):
    console_noise.restore_import_logging()
    console_noise.quieten_import_logging()
    try:
        # Each message on *its own* logger. A filter is a (logger, needle)
        # pair, so emitting every message on every logger would leave one
        # legitimately unfiltered - and a version of this test that did that
        # passed only by accident of logger propagation.
        for logger_name, needle, _ in console_noise.KNOWN_LOG_RECORDS:
            message = (REDIRECTS if needle in REDIRECTS else KERNEL_PREFERENCE)
            assert needle in message, f"no sample message matches {needle!r}"
            logging.getLogger(logger_name).warning(message)
        assert "Redirects are currently not supported" not in caplog.text
        assert "register_constant() on Enum subclasses" not in caplog.text
    finally:
        console_noise.restore_import_logging()


def test_other_records_from_the_same_loggers_survive():
    """A level change would have hidden these. A message filter must not.

    A handler is attached directly to each logger under test: torch configures
    propagation on these, so a root-level capture would prove nothing.
    """
    console_noise.restore_import_logging()
    console_noise.quieten_import_logging()

    seen = []

    class Recorder(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    handlers = []
    try:
        for logger_name, needle, _ in console_noise.KNOWN_LOG_RECORDS:
            logger = logging.getLogger(logger_name)
            handler = Recorder(level=logging.DEBUG)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
            handlers.append((logger, handler))

            logger.warning("a genuine problem here")
            logger.error("something actually broke")
            # Each logger filters only its own known message - the filters are
            # per-logger and per-message, not a shared blocklist.
            logger.warning(f"prefix {needle} suffix")

        assert seen.count("a genuine problem here") == 2
        assert seen.count("something actually broke") == 2
        for _, needle, _ in console_noise.KNOWN_LOG_RECORDS:
            assert not any(needle in message for message in seen), needle
    finally:
        for logger, handler in handlers:
            logger.removeHandler(handler)
            logger.setLevel(logging.NOTSET)
        console_noise.restore_import_logging()


def test_the_log_filters_are_removable():
    console_noise.restore_import_logging()
    console_noise.quieten_import_logging()
    installed = [logging.getLogger(name).filters
                 for name, _, _ in console_noise.KNOWN_LOG_RECORDS]
    assert all(filters for filters in installed)

    console_noise.restore_import_logging()
    for name, _, _ in console_noise.KNOWN_LOG_RECORDS:
        assert not any(isinstance(f, console_noise.KnownMessageFilter)
                       for f in logging.getLogger(name).filters)
    console_noise.quieten_import_logging()          # leave it as we found it


def test_installing_twice_does_not_stack_filters():
    console_noise.restore_import_logging()
    console_noise.quieten_import_logging()
    console_noise.quieten_import_logging()
    for name, _, _ in console_noise.KNOWN_LOG_RECORDS:
        matching = [f for f in logging.getLogger(name).filters
                    if isinstance(f, console_noise.KnownMessageFilter)]
        assert len(matching) == 1


def test_no_logger_level_is_changed():
    """Filtering by message, never by silencing a whole logger."""
    console_noise.restore_import_logging()
    before = {name: logging.getLogger(name).level
              for name, _, _ in console_noise.KNOWN_LOG_RECORDS}
    console_noise.quieten_import_logging()
    for name, level in before.items():
        assert logging.getLogger(name).level == level


def test_the_root_logger_is_left_alone():
    console_noise.restore_import_logging()
    before = list(logging.getLogger().filters)
    console_noise.quieten_import_logging()
    assert logging.getLogger().filters == before


def test_an_unformattable_record_is_let_through():
    """When in doubt, show it."""
    log_filter = console_noise.KnownMessageFilter("anything")
    record = logging.LogRecord("x", logging.WARNING, "f", 1,
                               "%d and %d", (1,), None)   # too few arguments
    assert log_filter.filter(record) is True


def test_the_environment_variable_disables_import_filtering(monkeypatch):
    console_noise.restore_import_logging()
    monkeypatch.setenv(console_noise.VERBOSE_IMPORT_ENVIRONMENT_VARIABLE, "1")
    console_noise.quieten_import_logging()
    assert console_noise._INSTALLED_LOG_FILTERS == []
    monkeypatch.delenv(console_noise.VERBOSE_IMPORT_ENVIRONMENT_VARIABLE)
    console_noise.quieten_import_logging()          # leave it as we found it


def test_a_real_import_is_quiet_but_the_escape_hatch_works():
    """The whole point, verified in a child process."""
    command = [sys.executable, "-c", "import delegate_doctor"]

    quiet = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "Redirects are currently not supported" not in quiet.stderr
    assert "register_constant()" not in quiet.stderr

    import os
    environment = {**os.environ,
                   console_noise.VERBOSE_IMPORT_ENVIRONMENT_VARIABLE: "1"}
    loud = subprocess.run(command, capture_output=True, text=True, check=True,
                          env=environment)
    assert "Redirects are currently not supported" in loud.stderr


# --- the policy is documented, not just implemented --------------------------

def test_every_suppressed_message_carries_a_reason():
    for _, _, why in console_noise.KNOWN_WARNINGS:
        assert why and len(why) > 20
    for _, _, why in console_noise.KNOWN_LOG_RECORDS:
        assert why and len(why) > 20


def test_the_policy_can_be_printed():
    text = console_noise.describe_policy()
    assert "LeafSpec" in text
    assert "cpuinfo_utils" in text
    assert "Redirects are currently not supported" in text
    assert "passed through untouched" in text


def test_the_allowlist_stays_short():
    """A growing list would mean the policy has stopped being an allowlist."""
    assert len(console_noise.KNOWN_WARNINGS) <= 4
    assert len(console_noise.KNOWN_LOG_RECORDS) <= 4
    assert len(console_noise.KNOWN_NATIVE_STDERR) <= 6


def test_there_is_no_blanket_suppression_anywhere_in_the_module():
    import ast

    source = open(console_noise.__file__).read()
    # Strip comments and the module docstring: both legitimately *name* the
    # things this module refuses to do.
    docstring = ast.get_docstring(ast.parse(source)) or ""
    code = "\n".join(line for line in source.replace(docstring, "").splitlines()
                     if not line.strip().startswith("#"))
    for forbidden in ('filterwarnings("ignore")', "simplefilter('ignore')",
                      'simplefilter("ignore")', "devnull", "logging.disable"):
        assert forbidden not in code, f"blanket suppression found: {forbidden}"
