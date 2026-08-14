"""Read the model file locally, before anything is considered for sending.

Two jobs, both deterministic:

  * **Understand what is there.** Classes, constructors, forward signatures,
    tensor-shaped literals and checkpoint references, found by parsing - the
    file is never executed to inspect it.

  * **Decide what may be sent.** Exactly the file the user named, scrubbed of
    credential material and home paths. Local imports are *identified* so the
    user can be asked about them individually; they are never bundled in.

Nothing here reaches the network, and nothing here executes user code.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from . import privacy

# A source file large enough to be a dataset is not a model definition, and
# would push everything useful out of the model's context anyway.
MAX_SOURCE_CHARACTERS = 60_000


class SourceInspectionError(RuntimeError):
    """The selected file cannot be inspected."""


@dataclass
class ModelCandidate:
    """One plausible model definition found in the file."""

    name: str
    kind: str                 # "class" or "instance"
    line: int = 0
    base_classes: list = field(default_factory=list)
    init_parameters: list = field(default_factory=list)
    forward_parameters: list = field(default_factory=list)

    @property
    def looks_like_module(self) -> bool:
        return any("Module" in base for base in self.base_classes)


@dataclass
class SourceFacts:
    """What local inspection established, with no AI involved."""

    path: Path
    source: str
    candidates: list = field(default_factory=list)
    local_imports: list = field(default_factory=list)
    tensor_literals: list = field(default_factory=list)
    checkpoint_references: list = field(default_factory=list)

    @property
    def module_candidates(self) -> list:
        return [c for c in self.candidates if c.looks_like_module]

    @property
    def unambiguous_model(self):
        """The single obvious model, or None if there is a choice to make."""
        modules = self.module_candidates
        return modules[0] if len(modules) == 1 else None


def _annotation_name(node) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def inspect_source(path: Path, source: str = None) -> SourceFacts:
    """Parse the model file and record what can be established for certain."""
    path = Path(path)
    if source is None:
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise SourceInspectionError(
                f"{path.name} is not readable as UTF-8 Python source.")
        except OSError as error:
            raise SourceInspectionError(f"Could not read {path.name}: {error.strerror}")

    if len(source) > MAX_SOURCE_CHARACTERS:
        raise SourceInspectionError(
            f"{path.name} is {len(source):,} characters, larger than the "
            f"{MAX_SOURCE_CHARACTERS:,} DelegateDoctor will inspect.\n"
            f"\n"
            f"Point it at the file that defines the model, or construct the "
            f"model in Python and call optimize() on it directly."
        )

    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise SourceInspectionError(
            f"Could not parse {path.name} as Python (line {error.lineno}): "
            f"{error.msg}")

    facts = SourceFacts(path=path, source=source)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            facts.candidates.append(_class_candidate(node))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            facts.local_imports.extend(_local_imports(node, path))
        elif isinstance(node, ast.Assign):
            facts.candidates.extend(_instance_candidates(node))

    for node in ast.walk(tree):
        facts.tensor_literals.extend(_tensor_literals(node))
        facts.checkpoint_references.extend(_checkpoint_references(node))

    # Stable order, and no duplicates, so prompts and questions are repeatable.
    facts.local_imports = sorted(set(facts.local_imports))
    facts.checkpoint_references = sorted(set(facts.checkpoint_references))
    return facts


def _class_candidate(node: ast.ClassDef) -> ModelCandidate:
    candidate = ModelCandidate(
        name=node.name, kind="class", line=node.lineno,
        base_classes=[_annotation_name(base) for base in node.bases],
    )
    for item in node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        parameters = [argument.arg for argument in item.args.args
                      if argument.arg != "self"]
        if item.name == "__init__":
            candidate.init_parameters = parameters
        elif item.name == "forward":
            candidate.forward_parameters = parameters
    return candidate


def _instance_candidates(node: ast.Assign) -> list:
    """`model = SomeNet(...)` at module level."""
    if not isinstance(node.value, ast.Call):
        return []
    found = []
    for target in node.targets:
        if isinstance(target, ast.Name):
            found.append(ModelCandidate(
                name=target.id, kind="instance", line=node.lineno,
                base_classes=[_annotation_name(node.value.func)],
            ))
    return found


def _local_imports(node, path: Path) -> list:
    """Imports that resolve to a file sitting beside the model source.

    Only siblings count. An installed package is not the user's code and is
    never a candidate for transmission.
    """
    names = []
    if isinstance(node, ast.ImportFrom):
        if node.level and node.level > 0:
            names.append(node.module or "")
        elif node.module:
            names.append(node.module.split(".")[0])
    else:
        for alias in node.names:
            names.append(alias.name.split(".")[0])

    local = []
    for name in names:
        if not name:
            continue
        sibling = path.parent / f"{name}.py"
        if sibling.is_file():
            local.append(sibling.name)
    return local


_TENSOR_FACTORIES = {"randn", "zeros", "ones", "empty", "rand", "randint"}


def _tensor_literals(node) -> list:
    """`torch.randn(1, 3, 224, 224)` - a strong hint at the real input shape."""
    if not isinstance(node, ast.Call):
        return []
    function = node.func
    if not isinstance(function, ast.Attribute) or function.attr not in _TENSOR_FACTORIES:
        return []
    shape = []
    for argument in node.args:
        if isinstance(argument, ast.Constant) and isinstance(argument.value, int):
            shape.append(argument.value)
        elif isinstance(argument, (ast.Tuple, ast.List)):
            for element in argument.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, int):
                    shape.append(element.value)
    return [{"generator": function.attr, "shape": shape}] if len(shape) >= 2 else []


def _checkpoint_references(node) -> list:
    """String literals that look like a local weights file."""
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        return []
    text = node.value
    if text.lower().endswith((".pth", ".pt", ".ckpt", ".safetensors", ".bin")):
        # Only the file name: a full path would disclose a directory layout.
        return [Path(text).name]
    return []


# --- what may leave the machine ---------------------------------------------


def prepare_source_for_transmission(facts: SourceFacts) -> str:
    """The exact text that would be sent, scrubbed.

    Credential material is redacted and home paths reduced to `~`, so a stray
    token in a comment or an absolute path in a default argument does not
    travel with the source the user agreed to send.
    """
    return privacy.sanitize_for_transmission(facts.source)


def transmission_is_safe(text: str) -> tuple:
    """(safe, reason). Refuses rather than sending something still credential-shaped."""
    if privacy.contains_secret(text):
        return False, (
            "The sanitized source still contains credential-shaped material.\n"
            "\n"
            "DelegateDoctor stopped rather than sending it. Remove the secret "
            "from the file, or construct the model in Python and call\n"
            "optimize() on it directly, which never sends source anywhere."
        )
    return True, ""


def summarize_for_console(facts: SourceFacts) -> str:
    """A short local-inspection summary, printed before anything is sent."""
    lines = [f"Source                  {facts.path.name}"]
    modules = facts.module_candidates
    if len(modules) == 1:
        lines.append(f"Model                   {modules[0].name}")
    elif modules:
        lines.append(f"Model candidates        "
                     f"{', '.join(c.name for c in modules)}")
    if facts.tensor_literals:
        shape = facts.tensor_literals[0]["shape"]
        lines.append(f"Input hint              {shape}")
    return "\n".join(lines)
