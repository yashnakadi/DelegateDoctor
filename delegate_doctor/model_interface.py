"""The DelegateDoctor model interface: how a `model.py` says what to analyze.

A model source file makes itself directly consumable by declaring two
functions:

    def delegate_doctor_model():
        model = resnet18(weights=None)
        model.eval()
        return model

    def delegate_doctor_inputs():
        return (torch.randn(1, 3, 224, 224),)

That is the whole contract. Two optional functions extend it:

    def delegate_doctor_kwargs():          -> dict
    def delegate_doctor_dynamic_shapes():  -> whatever torch.export accepts

When these exist, DelegateDoctor needs no AI to work out how to build the
model - the user already said. This is the deterministic path, and it is tried
before anything else is considered.

It is *not* an "exportable format": `model.py` is ordinary Python, not a
serialized `ExportedProgram`. Declaring the interface does not promise the
model exports; `torch.export` still decides that, and it decides in a child
process, because a user's model is a user's code.

Detection here is pure AST reading - the file is never imported to find out
whether it opted in. Only once the interface is found does anything execute,
and then only inside `sanitized_child_environment()`, which carries no AI
credential and no cloud secret.

    Trust boundary: running `model.py` is running the user's own Python, with
    their imports and their side effects. The child process keeps credentials
    out and keeps a crash out of DelegateDoctor's process. It is not an OS
    sandbox, and this file does not pretend otherwise.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .agent.privacy import redact, redact_home_paths, sanitized_child_environment

# The interface, named once.
MODEL_FUNCTION = "delegate_doctor_model"
INPUTS_FUNCTION = "delegate_doctor_inputs"
KWARGS_FUNCTION = "delegate_doctor_kwargs"
DYNAMIC_SHAPES_FUNCTION = "delegate_doctor_dynamic_shapes"

REQUIRED_FUNCTIONS = (MODEL_FUNCTION, INPUTS_FUNCTION)
OPTIONAL_FUNCTIONS = (KWARGS_FUNCTION, DYNAMIC_SHAPES_FUNCTION)

EXPORT_TIMEOUT_SECONDS = 600
MAX_FEEDBACK_CHARACTERS = 800

SUCCESS_MARKER = "DELEGATE_DOCTOR_EXPORT_OK"
FAILURE_MARKER = "DELEGATE_DOCTOR_EXPORT_FAILED"


# What went wrong, kept apart because the answers are different. "The weights
# could not be downloaded" is a resource problem; "torch.export refused this
# graph" is an incompatibility AI may be able to adjust around.
STAGE_IMPORT = "import"
STAGE_INTERFACE = "interface"
STAGE_CONSTRUCTION = "construction"
STAGE_FORWARD = "forward"
STAGE_EXPORT = "export"


class ModelInterfaceError(RuntimeError):
    """The interface is present but could not produce an ExportedProgram.

    Carries the structured `failure` when one was parsed, so a caller can show
    the real exception under `--verbose` instead of a generic sentence.
    """

    def __init__(self, message: str, failure=None):
        super().__init__(message)
        self.failure = failure


@dataclass(frozen=True)
class ExportFailure:
    """The real reason `torch.export` refused, kept structured.

    Parsed from labelled lines the adapter prints rather than scraped out of a
    traceback, so nothing here is a guess about text formatting.
    """

    stage: str = STAGE_EXPORT
    exception_type: str = ""
    message: str = ""
    traceback_text: str = ""
    kind: str = ""

    @property
    def is_export_stage(self) -> bool:
        """Did the model build and only the export refuse?

        The one failure AI assistance can plausibly help with. A missing
        dependency or unavailable weights is a fact about the environment.
        """
        return self.stage in (STAGE_EXPORT, STAGE_FORWARD)

    def summary(self) -> str:
        if self.exception_type and self.message:
            return f"{self.exception_type}: {self.message}"
        return self.exception_type or self.message or "export failed"

    def describe(self) -> str:
        """The verbose block. Sanitized; never secrets, weights or tensors."""
        lines = ["", "Export failure",
                 f"Stage                   {self.stage}",
                 f"Type                    {self.exception_type or 'unknown'}",
                 f"Message                 {self.message or '(none reported)'}"]
        if self.traceback_text:
            lines += ["", "Traceback", self.traceback_text]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"stage": self.stage, "type": self.exception_type,
                "message": self.message, "kind": self.kind}


@dataclass(frozen=True)
class InterfaceReport:
    """What the source file declares, read without importing it."""

    path: Path
    found_functions: tuple = ()
    missing_functions: tuple = ()
    parse_error: str = ""

    @property
    def complete(self) -> bool:
        """Are both required functions declared?"""
        return not self.missing_functions and not self.parse_error

    @property
    def partial(self) -> bool:
        """One required function but not the other - almost certainly a mistake."""
        return bool(self.found_functions) and bool(self.missing_functions)

    def describe(self) -> str:
        if self.complete:
            return (f"DelegateDoctor model interface  found "
                    f"({', '.join(self.found_functions)})")
        return "DelegateDoctor model interface  not found"


@dataclass
class PreparedModel:
    """A successful deterministic preparation, ready for the pipeline."""

    exported_program_path: Path
    inputs_path: Path
    model_name: str = ""
    source_path: Optional[Path] = None
    summary: str = ""
    # Named for symmetry with the AI path's outcome, so the CLI can hand either
    # to the same loader without asking which produced it.
    files_sent: list = field(default_factory=list)


def describe_interface() -> str:
    """The interface, as shown to a user who has not written it yet."""
    return (
        f"    def {MODEL_FUNCTION}():\n"
        f"        ...            # returns a torch.nn.Module\n"
        f"\n"
        f"    def {INPUTS_FUNCTION}():\n"
        f"        ...            # returns a tuple of example inputs\n"
    )


# --- detection: AST only, never an import ------------------------------------


def inspect_interface(path: Path, source: str = None) -> InterfaceReport:
    """Which interface functions `path` declares. Reads, never executes.

    Only module-level `def`s count. A function defined inside a class or
    another function is not importable under that name, so treating it as the
    interface would produce a confusing failure one step later.
    """
    path = Path(path)
    if source is None:
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as error:
            return InterfaceReport(path=path,
                                   missing_functions=REQUIRED_FUNCTIONS,
                                   parse_error=f"could not read: {error.strerror}")
        except UnicodeDecodeError:
            return InterfaceReport(path=path,
                                   missing_functions=REQUIRED_FUNCTIONS,
                                   parse_error="not UTF-8 text")

    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return InterfaceReport(path=path, missing_functions=REQUIRED_FUNCTIONS,
                               parse_error=f"syntax error on line {error.lineno}")

    declared = {node.name for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

    found = tuple(name for name in REQUIRED_FUNCTIONS + OPTIONAL_FUNCTIONS
                  if name in declared)
    missing = tuple(name for name in REQUIRED_FUNCTIONS if name not in declared)
    return InterfaceReport(path=path, found_functions=found,
                           missing_functions=missing)


# --- the adapter DelegateDoctor writes ---------------------------------------


def build_adapter_source(module_name: str, source_dir: Path,
                         output_path: Path, adjustment=None) -> str:
    """The child-process script. Contains no user text, only paths and names.

    Every validation the interface promises is enforced *here*, in the child,
    where the objects actually exist: a `nn.Module` return, a tuple of inputs,
    a dict of kwargs. A wrong return type becomes a named error rather than an
    obscure failure inside `torch.export`.

    A single Tensor from `delegate_doctor_inputs()` is normalized to a
    one-element tuple. That is the one convenience: returning `torch.randn(...)`
    for a single-input model is the obvious thing to write, and rejecting it
    would be pedantry. A list is also accepted and becomes a tuple - anything
    else is refused by name.
    """
    return f'''"""Generated by DelegateDoctor. Runs the model interface and exports.

Temporary, and deleted after the run. Contains no model source: it imports the
user's module by name and calls the two functions the interface defines.
"""

import sys
import traceback
from pathlib import Path

import torch

SOURCE_DIR = Path({str(source_dir)!r})
OUTPUT_PATH = Path({str(output_path)!r})

# Validated adjustments, if export assistance proposed any. Every value below
# came through schema validation and `repr()`, so it lands as data.
MODULE_ATTRIBUTES = {_attributes(adjustment)!r}
EXPORT_OPTIONS = {_export_options(adjustment)!r}
OUTPUT_INDEX = {_output_index(adjustment)!r}

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


def fail(message):
    print("{FAILURE_MARKER}")
    print(message)
    raise SystemExit(1)


def fail_stage(stage, marker, error):
    """A structured failure: which stage, which exception, and the traceback.

    Printed as labelled lines rather than left for the parent to scrape out of
    a traceback, so `--verbose` can show the real reason without guessing.
    """
    print("{FAILURE_MARKER}")
    print(marker + ":" + stage)
    print("STAGE: " + stage)
    print("TYPE: " + type(error).__name__)
    print("MESSAGE: " + str(error).strip().replace(chr(10), " | ")[:400])
    print("TRACEBACK:")
    traceback.print_exc(limit=8)
    raise SystemExit(1)


def call(user_module, name, required=True):
    function = getattr(user_module, name, None)
    if function is None:
        if required:
            fail("MISSING_FUNCTION:" + name)
        return None
    if not callable(function):
        fail("NOT_CALLABLE:" + name)
    try:
        return function()
    except TypeError as error:
        # The interface takes no arguments; a signature that needs them is a
        # contract error, not a model error, and deserves to say so.
        if "argument" in str(error):
            fail("TAKES_ARGUMENTS:" + name + ": " + str(error))
        fail("CALL_FAILED:" + name + "\\n" + traceback.format_exc(limit=6))
    except Exception:
        fail("CALL_FAILED:" + name + "\\n" + traceback.format_exc(limit=6))


class _SelectOutput(torch.nn.Module):
    """Keep one output of a multi-output model.

    Auxiliary classifier heads are the usual reason an otherwise fine model
    will not export: the forward returns a namedtuple, and only the primary
    logits matter for deployment. DelegateDoctor owns this wrapper; the AI only
    chooses the index, and only from a validated integer.
    """

    def __init__(self, inner, index):
        super().__init__()
        self.inner = inner
        self.index = index

    def forward(self, *args, **kwargs):
        result = self.inner(*args, **kwargs)
        if isinstance(result, (tuple, list)):
            return result[self.index]
        return result


def normalize_inputs(value):
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    fail("BAD_INPUTS:{INPUTS_FUNCTION}() returned "
         + type(value).__name__
         + "; expected a tuple of example inputs")


def main():
    try:
        user_module = __import__({module_name!r})
    except ModuleNotFoundError as error:
        missing = getattr(error, "name", "") or str(error)
        # The model's own module failing to import is a different problem from
        # one of its dependencies being absent, and conflating them would send
        # the user to install a package that is their own file.
        if missing == {module_name!r}:
            fail("SOURCE_NOT_IMPORTABLE:" + missing)
        fail("MISSING_DEPENDENCY:" + missing)
    except Exception:
        fail("IMPORT_FAILED:\\n" + traceback.format_exc(limit=6))

    model = call(user_module, {MODEL_FUNCTION!r})
    if not isinstance(model, torch.nn.Module):
        fail("BAD_MODEL:{MODEL_FUNCTION}() returned "
             + type(model).__name__
             + "; expected a torch.nn.Module")

    args = normalize_inputs(call(user_module, {INPUTS_FUNCTION!r}))

    kwargs = call(user_module, {KWARGS_FUNCTION!r}, required=False) or {{}}
    if not isinstance(kwargs, dict):
        fail("BAD_KWARGS:{KWARGS_FUNCTION}() returned "
             + type(kwargs).__name__ + "; expected a dict")
    if any(not isinstance(key, str) for key in kwargs):
        fail("BAD_KWARGS:{KWARGS_FUNCTION}() returned non-string keys")

    dynamic_shapes = call(user_module, {DYNAMIC_SHAPES_FUNCTION!r}, required=False)

    # torch.export needs inference mode, and the interface documents that the
    # returned model is exported in eval mode. Doing it here rather than
    # trusting the user to is one less way to get a silently different graph.
    model.eval()

    # Applied to the model the user's own factory built - the interface stays
    # authoritative, and this only adjusts it.
    for attribute_name, attribute_value in MODULE_ATTRIBUTES.items():
        if not hasattr(model, attribute_name):
            fail("UNKNOWN_ATTRIBUTE:" + attribute_name)
        setattr(model, attribute_name, attribute_value)

    if OUTPUT_INDEX is not None:
        model = _SelectOutput(model, OUTPUT_INDEX)

    try:
        with torch.no_grad():
            model(*args, **kwargs)
    except Exception as error:
        fail_stage("forward", "FORWARD_FAILED", error)

    export_options = {{}}
    if dynamic_shapes is not None:
        export_options["dynamic_shapes"] = dynamic_shapes
    export_options.update(EXPORT_OPTIONS)

    try:
        exported = torch.export.export(model, args=args, kwargs=kwargs,
                                       **export_options)
    except Exception as error:
        fail_stage("export", "EXPORT_FAILED", error)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported, OUTPUT_PATH)
    torch.save((args, kwargs), OUTPUT_PATH.with_suffix(".inputs.pt"))
    print("{SUCCESS_MARKER}")


main()
'''


# --- running it ---------------------------------------------------------------


def _sanitize(text: str) -> str:
    """A short, path-free, secret-free account of what the child reported."""
    import re

    cleaned = redact_home_paths(redact(text or ""))
    cleaned = re.sub(r'File "[^"]*[/\\]([^"/\\]+)"', r'File "\1"', cleaned)
    return cleaned.strip()[:MAX_FEEDBACK_CHARACTERS]


def run_adapter(adapter_path: Path, working_dir: Path,
                timeout: int = EXPORT_TIMEOUT_SECONDS) -> tuple:
    """Execute the adapter in a sanitized child process. Returns (ok, output)."""
    try:
        completed = subprocess.run(
            [sys.executable, str(adapter_path)],
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=sanitized_child_environment(),
        )
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT: export did not finish within {timeout}s"

    output = (completed.stdout or "") + (completed.stderr or "")
    ok = completed.returncode == 0 and SUCCESS_MARKER in (completed.stdout or "")
    return ok, output


def prepare_from_interface(model_path: Path, workspace: Path,
                           announce=print, adjustment=None,
                           quiet: bool = False) -> PreparedModel:
    """Run the declared interface and export. Deterministic: no AI, no network.

    Raises `ModelInterfaceError` when the interface is absent or when export
    fails - the caller decides whether AI is worth offering next, because that
    is a policy question and this module only knows mechanics.
    """
    # Resolved before anything else: the child inserts this file's directory
    # into sys.path *and* runs with that directory as its cwd, so a relative
    # path would be applied twice and the module would not be importable.
    model_path = Path(model_path).resolve()
    workspace = Path(workspace)

    report = inspect_interface(model_path)
    if not report.complete:
        raise ModelInterfaceError(missing_interface_message(report))

    if not quiet:
        announce("DelegateDoctor model interface  found")
        announce("Exporting...")

    workspace.mkdir(parents=True, exist_ok=True)
    output_path = workspace / "prepared_model.pt2"
    adapter_path = workspace / "run_model_interface.py"
    adapter_path.write_text(build_adapter_source(
        model_path.stem, model_path.parent, output_path, adjustment),
        encoding="utf-8")

    try:
        ok, output = run_adapter(adapter_path, model_path.parent)
    finally:
        # DelegateDoctor's generated file has served its purpose. Leaving code
        # lying around next to a user's model is nobody's idea of tidy.
        adapter_path.unlink(missing_ok=True)

    if not ok or not output_path.is_file():
        failure = parse_failure(output)
        raise ModelInterfaceError(
            explain_failure(output, model_path), failure=failure)

    if not quiet:
        announce("PyTorch export                  PASS")
    return PreparedModel(
        exported_program_path=output_path,
        inputs_path=output_path.with_suffix(".inputs.pt"),
        model_name=model_path.stem,
        source_path=model_path,
        summary=f"prepared from the model interface in {model_path.name}",
    )


# --- messages ------------------------------------------------------------------


def missing_interface_message(report: InterfaceReport) -> str:
    """Why the deterministic path could not run, and what to write instead."""
    if report.parse_error:
        headline = (f"{report.path.name} could not be read as Python "
                    f"({report.parse_error}).")
    elif report.partial:
        headline = (f"{report.path.name} defines "
                    f"{', '.join(report.found_functions)} but not "
                    f"{', '.join(report.missing_functions)}.")
    else:
        headline = "DelegateDoctor model interface not found."

    return (
        f"{headline}\n"
        f"\n"
        f"Add both functions to {report.path.name}:\n"
        f"\n"
        f"{describe_interface()}"
    )


# What the child printed, translated into something worth reading. The marker
# prefixes come from the adapter above, so this table is not guesswork.
_FAILURE_EXPLANATIONS = {
    "SOURCE_NOT_IMPORTABLE": (
        "The model source itself could not be imported by name.\n"
        "\n"
        "This usually means the filename is not a valid Python module name -\n"
        "a hyphen, a leading digit, or a name that shadows a stdlib module."
    ),
    "MISSING_DEPENDENCY": (
        "The model source imports a package that is not installed.\n"
        "\n"
        "DelegateDoctor never installs dependencies on your behalf. Install it\n"
        "into this environment and run the same command again."
    ),
    "BAD_MODEL": (
        f"{MODEL_FUNCTION}() must return a torch.nn.Module."
    ),
    "BAD_INPUTS": (
        f"{INPUTS_FUNCTION}() must return a tuple of positional example\n"
        f"inputs. A single Tensor is accepted and treated as a one-element\n"
        f"tuple; anything else is refused."
    ),
    "BAD_KWARGS": (
        f"{KWARGS_FUNCTION}() must return a dict with string keys."
    ),
    "TAKES_ARGUMENTS": (
        "The interface functions take no arguments."
    ),
    "UNKNOWN_ATTRIBUTE": (
        "The proposed adjustment named an attribute the model does not have."
    ),
    "MISSING_FUNCTION": (
        "The function was declared in the file but could not be imported.\n"
        "It must be defined at module level."
    ),
    "NOT_CALLABLE": (
        "The interface name exists but is not a function."
    ),
    "FORWARD_FAILED": (
        "The model raised when called with the inputs the interface returned.\n"
        "\n"
        f"Check that {INPUTS_FUNCTION}() matches what the model expects."
    ),
    "EXPORT_FAILED": (
        "torch.export could not capture this model."
    ),
    "IMPORT_FAILED": (
        "Importing the model source raised before anything could be built."
    ),
    "CALL_FAILED": (
        "An interface function raised."
    ),
}


def _attributes(adjustment) -> dict:
    return dict(getattr(adjustment, "module_attributes", {}) or {})


def _export_options(adjustment) -> dict:
    return dict(getattr(adjustment, "export_options", {}) or {})


def _output_index(adjustment):
    return getattr(adjustment, "output_index", None)


def parse_failure(output: str) -> ExportFailure:
    """Read the adapter's labelled failure lines back into a value.

    Falls back to the marker kind when the structured lines are absent, so an
    older-shaped failure still names a stage rather than nothing.
    """
    text = output or ""
    fields = {}
    traceback_lines = []
    collecting = False

    for line in text.splitlines():
        if collecting:
            traceback_lines.append(line)
            continue
        if line.startswith("TRACEBACK:"):
            collecting = True
            continue
        for label in ("STAGE", "TYPE", "MESSAGE"):
            if line.startswith(f"{label}: "):
                fields[label.lower()] = line[len(label) + 2:].strip()

    kind = failure_kind(text)
    stage = fields.get("stage") or _STAGE_FOR_KIND.get(kind, STAGE_EXPORT)

    # Import and interface failures are reported through the simpler `fail()`,
    # which prints `MARKER:payload` rather than labelled lines. Using that
    # payload keeps `--verbose` useful for every category rather than only the
    # ones that go through `fail_stage`.
    message = fields.get("message", "")
    exception_type = fields.get("type", "")

    if not message and kind:
        for line in text.splitlines():
            if line.startswith(f"{kind}:"):
                message = line[len(kind) + 1:].strip()
                break

    # `fail()` prints a raw traceback rather than labelled lines, so the real
    # exception is its last line. Without this, `--verbose` showed the failing
    # *function's* name where the user needed the reason it failed.
    if not exception_type:
        exception_type, detail = _exception_from_traceback(text)
        if detail:
            message = f"{message}: {detail}" if message else detail
        exception_type = exception_type or kind

    return ExportFailure(
        stage=stage,
        exception_type=exception_type or kind,
        message=message,
        # Sanitized before it is stored, not before it is shown: nothing that
        # reaches a report or a provider has home paths or secrets in it.
        traceback_text=_sanitize("\n".join(traceback_lines)),
        kind=kind,
    )


def _exception_from_traceback(text: str) -> tuple:
    """(type, message) from the last `Type: message` line of a traceback."""
    for line in reversed((text or "").splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith(("File ", "Traceback", "^", "~")):
            continue
        head, separator, tail = stripped.partition(": ")
        if separator and head and head[0].isupper() and " " not in head:
            return head, tail.strip()[:300]
    return "", ""


_STAGE_FOR_KIND = {
    "MISSING_DEPENDENCY": STAGE_IMPORT,
    "SOURCE_NOT_IMPORTABLE": STAGE_IMPORT,
    "IMPORT_FAILED": STAGE_IMPORT,
    "MISSING_FUNCTION": STAGE_INTERFACE,
    "NOT_CALLABLE": STAGE_INTERFACE,
    "TAKES_ARGUMENTS": STAGE_INTERFACE,
    "BAD_MODEL": STAGE_CONSTRUCTION,
    "BAD_INPUTS": STAGE_INTERFACE,
    "BAD_KWARGS": STAGE_INTERFACE,
    "CALL_FAILED": STAGE_CONSTRUCTION,
    "UNKNOWN_ATTRIBUTE": STAGE_CONSTRUCTION,
    "FORWARD_FAILED": STAGE_FORWARD,
    "EXPORT_FAILED": STAGE_EXPORT,
}


def failure_kind(output: str) -> str:
    """Which adapter marker the child reported, or "" when none did.

    The banner line is removed first. `DELEGATE_DOCTOR_EXPORT_FAILED` contains
    the substring `EXPORT_FAILED`, so scanning the raw text classified every
    failure - a missing dependency, a bad return type, an unavailable
    checkpoint - as an export failure, which is what collapsed genuinely
    different problems into one answer.
    """
    text = "\n".join(line for line in (output or "").splitlines()
                     if line.strip() != FAILURE_MARKER)
    for marker in _FAILURE_EXPLANATIONS:
        if marker in text:
            return marker
    return ""


def is_export_failure(output: str) -> bool:
    """Did the interface run correctly and `torch.export` refuse the model?

    This is the one failure worth offering AI preparation for. A missing
    dependency or a wrong return type is a fact about the file that no
    provider can change, and offering to send it away would waste a request.
    """
    return failure_kind(output) in ("EXPORT_FAILED", "FORWARD_FAILED")


def explain_failure(output: str, model_path: Path) -> str:
    """Turn the child's marker and traceback into an actionable message."""
    kind = failure_kind(output)
    detail = _sanitize(output)
    explanation = _FAILURE_EXPLANATIONS.get(
        kind, "The model interface did not produce an ExportedProgram.")

    return (
        f"DETERMINISTIC PREPARATION FAILED\n"
        f"\n"
        f"{model_path.name} declares the DelegateDoctor model interface, but\n"
        f"it did not produce an ExportedProgram.\n"
        f"\n"
        f"{explanation}\n"
        f"\n"
        f"What the export reported:\n"
        f"{detail}"
    )


def model_spec_from_prepared(prepared: PreparedModel, name: str = "") -> object:
    """Load what deterministic preparation produced into the pipeline's ModelSpec.

    The same shape as the AI path's loader, and deliberately the same artifact
    contract, so everything downstream is identical whichever path got here.
    """
    import torch

    from .export_model import ModelSpec

    program = torch.export.load(str(prepared.exported_program_path))
    args, kwargs = torch.load(str(prepared.inputs_path), map_location="cpu",
                              weights_only=True)
    return ModelSpec(
        name=name or prepared.model_name or "PyTorch Model",
        exported_program=program,
        example_args=tuple(args),
        example_kwargs=dict(kwargs or {}),
        description=prepared.summary or "prepared from the model interface",
    )
