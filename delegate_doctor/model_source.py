"""Work out what the user pointed DelegateDoctor at, and where it lives.

One kind of input reaches the CLI:

    delegate-doctor optimize model.py    PyTorch model source

and one convenience: a *bare* filename with no directory part is also looked for
in `models/`, the local workspace beside the repository. That is one documented
lookup, not a search - `models/` is checked and nothing else.

    optimize model.py             -> ./model.py, else ./models/model.py
    optimize models/model.py      -> exactly that
    optimize projects/a/model.py  -> exactly that

An explicit path always wins and is never second-guessed, so a user who typed a
directory always gets the file they named or a clear error about that file.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# The local workspace for a user's own models. Git-ignored; `examples/` stays
# the checked-in demonstration suite and is never written to.
WORKSPACE_DIR_NAME = "models"

PYTHON_SUFFIX = ".py"

# Recognised only to be refused with an explanation. Serialized programs and
# input tuples were once accepted here; they are now internal artifacts, and a
# user who learned the old command deserves better than "unsupported file".
EXPORTED_SUFFIX = ".pt2"
INPUTS_SUFFIX = ".pt"

_URL_SCHEMES = ("http://", "https://", "git://", "git@", "ssh://", "ftp://",
                "file://")
_BARE_HOST = re.compile(r"^(www\.)?[\w-]+(\.[\w-]+)+/")


class ModelSourceError(RuntimeError):
    """The supplied path is not something DelegateDoctor can analyze."""


@dataclass(frozen=True)
class ResolvedInput:
    """A validated local file, and how it was found."""

    path: Path
    kind: str                 # always "python"
    from_workspace: bool      # True when the models/ fallback supplied it

    @property
    def is_python(self) -> bool:
        return self.kind == "python"

    def describe_location(self) -> str:
        if self.from_workspace:
            return f"{self.path}  (found in {WORKSPACE_DIR_NAME}/)"
        return str(self.path)


def looks_like_url(target: str) -> bool:
    """Is this a remote address rather than a local path?"""
    text = (target or "").strip()
    if text.lower().startswith(_URL_SCHEMES):
        return True
    return bool(_BARE_HOST.match(text)) and not os.path.exists(text)


UNSUPPORTED_INPUT_MESSAGE = (
    "unsupported model input: DelegateDoctor analyzes a local PyTorch model\n"
    "\n"
    "  delegate-doctor optimize model.py    your model source\n"
    "\n"
    "For a model you already have in Python, use the API - it takes the live\n"
    "module and needs no files:\n"
    "\n"
    "  from delegate_doctor import optimize\n"
    "  result = optimize(model, args=(example_input,))\n"
    "\n"
    "Remote sources - repository URLs, any http(s) address - are not supported."
)

# What to say when someone types the entry point that used to exist. Naming the
# replacement matters more than naming the removal.
SERIALIZED_INPUT_MESSAGE = (
    "DelegateDoctor no longer takes serialized artifacts on the command line.\n"
    "\n"
    "Analyze the model in Python instead. The API accepts anything\n"
    "torch.export.export() can capture, and never uses AI:\n"
    "\n"
    "    import torch\n"
    "    from delegate_doctor import optimize\n"
    "\n"
    "    result = optimize(model.eval(), args=(example_input,))\n"
    "    print(result.summary)\n"
    "\n"
    "Or point the CLI at the model source, which it will prepare for you:\n"
    "\n"
    "    delegate-doctor optimize model.py"
)


def workspace_directory(workspace_root: Path | str = ".") -> Path:
    """Where a user's own model files are looked for."""
    return Path(workspace_root) / WORKSPACE_DIR_NAME


def has_directory_component(target: str) -> bool:
    """Did the user type a path, or just a filename?

    Anything with a separator - or an absolute path, or one starting with `.` -
    is treated as explicit and never redirected to the workspace.
    """
    text = str(target)
    if os.path.isabs(text):
        return True
    if text.startswith(("." + os.sep, ".." + os.sep, "./", "../", "~")):
        return True
    # Accept either separator, so a Windows-style path typed on any host is
    # still recognised as explicit rather than treated as a bare name.
    return os.sep in text or "/" in text or "\\" in text


def candidate_paths(target: str, workspace_root: Path | str = ".") -> list:
    """The paths that will be tried, in order. Exactly one or two of them."""
    direct = Path(target).expanduser()
    if has_directory_component(target):
        return [direct]
    # A bare filename: the current directory first, then the workspace.
    return [direct, workspace_directory(workspace_root) / direct.name]


def _classify(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == PYTHON_SUFFIX:
        return "python"
    return ""


def _check_readable_file(path: Path, label: str) -> None:
    """Shared checks once a path has been chosen."""
    if path.is_dir():
        raise ModelSourceError(
            f"{label} is a directory: {path}\n"
            f"\nDelegateDoctor takes a single file, not a directory."
        )
    # is_file() is a regular-file test, so device nodes, FIFOs and sockets are
    # rejected here rather than blocking forever inside a read.
    if not path.is_file():
        raise ModelSourceError(
            f"{label} is not a regular file: {path}\n"
            f"\nDevices, pipes and sockets are not supported."
        )


def resolve_model_input(target: str, workspace_root: Path | str = ".") -> ResolvedInput:
    """Find the model file the user meant, or explain precisely why not."""
    if not target or not str(target).strip():
        raise ModelSourceError(f"No model given.\n\n{UNSUPPORTED_INPUT_MESSAGE}")

    if looks_like_url(target):
        raise ModelSourceError(UNSUPPORTED_INPUT_MESSAGE)

    text = str(target)

    if text.lower().endswith(".pte"):
        raise ModelSourceError(
            f"That is an ExecuTorch artifact, not an input: {text}\n"
            f"\n"
            f"  .py   PyTorch model source - what DelegateDoctor takes\n"
            f"  .pte  ExecuTorch deployment artifact - DelegateDoctor's *output*\n"
            f"\n"
            f"A .pte cannot be optimized: its delegated regions are already\n"
            f"compiled blobs. Point DelegateDoctor at the model instead."
        )

    if text.lower().endswith((EXPORTED_SUFFIX, INPUTS_SUFFIX)):
        raise ModelSourceError(
            f"{text} is a serialized artifact, not a model source.\n"
            f"\n"
            f"{SERIALIZED_INPUT_MESSAGE}"
        )

    kind = _classify(Path(text))
    if not kind:
        raise ModelSourceError(
            f"Unsupported model file: {text}\n"
            f"\n"
            f"Expected a {PYTHON_SUFFIX} model source.\n"
            f"\n"
            f"{UNSUPPORTED_INPUT_MESSAGE}"
        )

    tried = candidate_paths(text, workspace_root)
    for index, candidate in enumerate(tried):
        if candidate.exists():
            _check_readable_file(candidate, "Model file")
            return ResolvedInput(path=candidate.resolve(), kind=kind,
                                 from_workspace=index > 0)

    raise ModelSourceError(_not_found_message("Model file", text, tried))


def _not_found_message(label: str, target: str, tried: list) -> str:
    text = f"{label} not found: {target}\n\nLooked in:\n"
    text += "\n".join(f"  {candidate}" for candidate in tried)
    if len(tried) > 1:
        text += (
            f"\n\nPut your model in {WORKSPACE_DIR_NAME}/ and run it by name:\n"
            f"\n"
            f"  {WORKSPACE_DIR_NAME}/model.py\n"
            f"  delegate-doctor optimize model.py\n"
            f"\n"
            f"or give an explicit path."
        )
    return text
