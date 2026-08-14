"""The examples are ordinary users of the public API, and the core knows nothing.

This is the architectural claim the whole demo story rests on: DelegateDoctor
has no U-Net, PSPNet or GhostNet code in it. These tests read the example files
and the package source rather than importing them, so the suite stays offline
and needs neither `segmentation_models_pytorch`, `timm` nor `torchvision`.
"""

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
PACKAGE_DIR = PROJECT_ROOT / "delegate_doctor"

# The demonstration workloads, discovered from the tree rather than listed.
# Examples live in purpose-named subdirectories (dd001_softmax/, dd003_avgpool/,
# fully_delegated/, ...), so a new example is a new file and not a test edit as
# well - which is what a hardcoded flat list used to make it.
#
# Two shapes are allowed, and every example is one or the other:
#
#   interface example   declares delegate_doctor_model() and
#                       delegate_doctor_inputs(), so
#                       `delegate-doctor optimize examples/<dir>/<name>.py`
#                       analyzes it with no AI involved. This is the normal
#                       shape and the documented onboarding path.
#   plain model source  declares neither, on purpose: it demonstrates what a
#                       model file looks like *before* the interface is added,
#                       which is the case optional AI preparation exists for.
#
# The shape is detected rather than declared, so converting an example between
# them is a one-file change.
DEMO_EXAMPLES = sorted(
    str(path.relative_to(EXAMPLES_DIR))
    for path in EXAMPLES_DIR.rglob("*.py")
    if "__pycache__" not in path.parts
)


# Architecture names that must not appear anywhere in the core package.
# Modules that legitimately name a library without knowing any architecture.
# `environment_check` reports whether TorchVision is installed; that is a
# packaging fact, not a model DelegateDoctor has code for.
ARCHITECTURE_NAME_EXEMPT = ("environment_check.py",)

ARCHITECTURE_NAMES = (
    "unet", "Unet", "U-Net", "unetplusplus", "UnetPlusPlus",
    "pspnet", "PSPNet", "deeplabv3", "DeepLabV3", "linknet", "Linknet",
    "ghostnet", "GhostNet", "mobilenet", "MobileNet",
    # The models DD-003 was validated on. A rule that generalizes cannot name
    # them, and the README claims exactly that, so the claim is guarded here.
    "inception", "Inception", "densenet", "DenseNet",
    "convnext", "ConvNeXt", "resnext", "ResNeXt",
    "segmentation_models_pytorch", "timm", "torchvision",
)


def code_without_docstrings(path: Path) -> str:
    """Module source with every docstring and comment removed.

    These files describe what they demonstrate, so a plain text scan cannot
    tell an explanation of a repair rule from an implementation of one.
    """
    import io
    import tokenize

    kept = []
    previous = tokenize.INDENT
    with open(path) as handle:
        for token in tokenize.generate_tokens(handle.readline):
            if token.type == tokenize.COMMENT:
                continue
            if token.type == tokenize.STRING and previous in (
                    tokenize.INDENT, tokenize.NEWLINE, tokenize.NL,
                    tokenize.DEDENT):
                previous = token.type
                continue
            if token.type not in (tokenize.NL, tokenize.NEWLINE):
                previous = token.type
            kept.append(token.string)
    return "\n".join(kept)


def example_source(name: str) -> str:
    path = EXAMPLES_DIR / name
    assert path.is_file(), f"missing example: {name}"
    return path.read_text()


def module_functions(name: str) -> set:
    return {node.name for node in ast.parse(example_source(name)).body
            if isinstance(node, ast.FunctionDef)}


def is_interface_example(name: str) -> bool:
    """Does this example declare the DelegateDoctor model interface?"""
    return {"delegate_doctor_model", "delegate_doctor_inputs"} <= \
        module_functions(name)


INTERFACE_EXAMPLES = [name for name in DEMO_EXAMPLES if is_interface_example(name)]
PLAIN_SOURCE_EXAMPLES = [name for name in DEMO_EXAMPLES
                         if not is_interface_example(name)]


# --- every demo example exists and parses -----------------------------------

@pytest.mark.parametrize("name", DEMO_EXAMPLES)
def test_the_example_exists_and_parses(name):
    ast.parse(example_source(name))


def test_examples_are_grouped_by_purpose():
    """Every example sits in a subdirectory naming what it demonstrates.

    The grouping is the point: a reader looking for the DD-003 evidence should
    find it without reading fourteen files.
    """
    for name in DEMO_EXAMPLES:
        assert "/" in name, f"{name} is not in a purpose-named subdirectory"


@pytest.mark.parametrize("name", INTERFACE_EXAMPLES)
def test_the_interface_example_declares_both_required_functions(name):
    """`delegate-doctor optimize examples/<dir>/<name>.py` must work with no AI."""
    from delegate_doctor import model_interface

    report = model_interface.inspect_interface(EXAMPLES_DIR / name)
    assert report.complete, f"{name} does not declare a usable interface"


@pytest.mark.parametrize("name", INTERFACE_EXAMPLES)
def test_the_interface_functions_take_no_arguments(name):
    """The contract is `f()`, so a signature needing arguments is a bug."""
    tree = ast.parse(example_source(name))
    for node in tree.body:
        if (isinstance(node, ast.FunctionDef)
                and node.name.startswith("delegate_doctor_")):
            arguments = node.args
            assert not arguments.args and not arguments.kwonlyargs, (
                f"{name}: {node.name}() takes arguments")


@pytest.mark.parametrize("name", PLAIN_SOURCE_EXAMPLES)
def test_the_plain_source_example_really_lacks_the_interface(name):
    """Its whole purpose is to be the case the interface is missing from.

    If someone helpfully adds `delegate_doctor_model()` to it, the AI
    preparation path loses its demonstration and this test says so.
    """
    from delegate_doctor import model_interface

    report = model_interface.inspect_interface(EXAMPLES_DIR / name)
    assert not report.complete, (
        f"{name} now declares the interface, so it no longer demonstrates "
        f"preparation without one")
    # It must still be a real model file, not an empty placeholder.
    tree = ast.parse(example_source(name))
    assert any(isinstance(node, (ast.ClassDef, ast.FunctionDef))
               for node in tree.body), f"{name} defines no model"


def test_at_least_one_example_of_each_shape_ships():
    """Both documented routes have a demonstration."""
    assert INTERFACE_EXAMPLES, "no example demonstrates the model interface"
    assert PLAIN_SOURCE_EXAMPLES, "no example demonstrates a file without one"


# --- no example reaches past the public API ---------------------------------

PRIVATE_ENTRY_POINTS = (
    "run_optimization", "pipeline", "dd001_softmax", "dd002_noop_alias",
    "export_model", "ModelSpec", "analyze_exported_program", "verify_repair",
    "decide_repair", "build_model_spec", "delegate_doctor.models",
)


@pytest.mark.parametrize("name", DEMO_EXAMPLES)
def test_the_example_does_not_use_a_private_entry_point(name):
    """An example that reached inside would not be testing the public surface."""
    source = example_source(name)
    code = "\n".join(line for line in source.splitlines()
                     if not line.strip().startswith("#"))
    # Strip the module docstring: prose may legitimately mention internals.
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree) or ""
    for token in PRIVATE_ENTRY_POINTS:
        in_code = token in code.replace(docstring, "")
        assert not in_code, f"{name} reaches past the public API: {token}"


@pytest.mark.parametrize("name", DEMO_EXAMPLES)
def test_the_example_contains_no_repair_rule_logic(name):
    """Examples construct models. They never implement or name a DD rule.

    Every docstring is stripped, not just the module's: an example may
    legitimately explain in prose which rule its fallback demonstrates, and
    that is documentation rather than logic.

    The tokens checked are the ones that can only be rule internals. A model
    containing a softmax or an alias is *the point* of several of these files,
    so those are not on the list - a naive text match could not tell a model
    that has the pattern from code that knows how to repair it.
    """
    code = code_without_docstrings(EXAMPLES_DIR / name)
    for token in ("DD-001", "DD-002", "aten.alias", "aten._softmax",
                  "detect(", "matches_portable_kernel"):
        assert token not in code, f"{name} contains rule-specific code: {token}"


@pytest.mark.parametrize("name", DEMO_EXAMPLES)
def test_the_example_does_not_build_its_own_report(name):
    """One report implementation, used by everything."""
    source = example_source(name)
    for token in ("html_report", "<html", "<!DOCTYPE", "generate_html_report",
                  "webbrowser"):
        assert token not in source, f"{name} duplicates report logic: {token}"


# --- nothing anywhere uses emoji --------------------------------------------

def _has_pictograph(text: str):
    for character in text:
        code = ord(character)
        if (0x1F000 <= code <= 0x1FAFF or 0x2600 <= code <= 0x27BF
                or code in (0xFE0F, 0x2705, 0x274C, 0x2728)):
            return character
    return None


def test_no_source_file_contains_emoji():
    """Typography and colour carry meaning here, not pictographs."""
    targets = list(PACKAGE_DIR.rglob("*.py")) + list(EXAMPLES_DIR.glob("*.py"))
    targets += list((PROJECT_ROOT / "tests").glob("*.py"))
    targets.append(PROJECT_ROOT / "README.md")
    for path in targets:
        found = _has_pictograph(path.read_text(encoding="utf-8"))
        assert found is None, f"{path.name} contains {found!r}"


# --- the core package has no architecture registry --------------------------

def test_the_demo_model_registry_is_gone():
    assert not (PACKAGE_DIR / "models.py").exists(), \
        "delegate_doctor/models.py still exists"


def test_the_package_does_not_export_a_model_catalog():
    import delegate_doctor

    for attribute in ("models", "MODEL_NAMES", "BUILTIN_EXAMPLES",
                      "build_model_spec", "create_model", "DISPLAY_NAMES"):
        assert not hasattr(delegate_doctor, attribute), \
            f"delegate_doctor still exposes {attribute}"


def strip_prose(source: str) -> str:
    """Source with comments and docstrings removed.

    The claim under test is that no core module contains architecture-specific
    *code*. A docstring recording where a pattern was first observed - DD-002
    citing timm's GhostNet, for instance - is evidence, not dispatch, and is
    worth keeping.
    """
    import io
    import tokenize

    kept = []
    previous_type = tokenize.INDENT
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            continue
        # A string that stands alone as a statement is a docstring.
        if token.type == tokenize.STRING and previous_type in (
                tokenize.INDENT, tokenize.NEWLINE, tokenize.NL, tokenize.DEDENT):
            previous_type = token.type
            continue
        if token.type not in (tokenize.NL, tokenize.NEWLINE):
            previous_type = token.type
        kept.append(token.string)
    return "\n".join(kept)


def test_no_core_module_names_a_demo_architecture():
    """The central claim: DelegateDoctor has no code for these models."""
    offenders = []
    for path in PACKAGE_DIR.rglob("*.py"):
        if path.name in ARCHITECTURE_NAME_EXEMPT:
            continue
        code = strip_prose(path.read_text())
        for name in ARCHITECTURE_NAMES:
            if name in code:
                offenders.append(f"{path.name}: {name}")
    assert offenders == [], f"core package has architecture-specific code: {offenders}"


def test_no_core_module_imports_a_demo_only_library():
    for path in PACKAGE_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            for name in names:
                assert name not in ("segmentation_models_pytorch", "timm",
                                    "torchvision"), \
                    f"{path.name} imports demo-only library {name}"


def test_the_package_has_no_model_name_dispatch():
    """No `if name == "unet": ...` anywhere in the core."""
    for path in PACKAGE_DIR.rglob("*.py"):
        source = path.read_text()
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Compare):
                continue
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and isinstance(
                        comparator.value, str):
                    assert comparator.value.lower() not in (
                        "unet", "fpn", "pspnet", "linknet", "ghostnet",
                        "deeplabv3plus", "unetplusplus",
                    ), f"{path.name} dispatches on a model name"


# --- the CLI no longer has a demo command -----------------------------------

def test_the_doctor_command_is_gone():
    from delegate_doctor import cli

    for name in ("run_doctor", "load_model_spec", "BUILTIN_EXAMPLES"):
        assert not hasattr(cli, name), f"cli still defines {name}"


@pytest.mark.parametrize("argv", [
    ["doctor", "unet"],
    ["doctor", "ghostnet"],
    ["doctor", "anything"],
    ["doctor"],
])
def test_the_doctor_subcommand_no_longer_resolves(argv):
    from delegate_doctor import cli

    with pytest.raises(SystemExit) as caught:
        cli.main(argv)
    assert caught.value.code != 0


def test_the_cli_help_offers_only_the_two_real_commands(capsys):
    from delegate_doctor import cli

    with pytest.raises(SystemExit):
        cli.main(["--help"])
    text = capsys.readouterr().out
    assert "optimize" in text
    assert "setup-android" in text
    assert "doctor MODEL" not in text
    for name in ("unet", "pspnet", "ghostnet", "linknet"):
        assert name not in text.replace("examples/unet.py", ""), \
            f"CLI help still advertises {name}"


def test_the_cli_help_points_at_the_python_api(capsys):
    from delegate_doctor import cli

    with pytest.raises(SystemExit):
        cli.main(["--help"])
    text = capsys.readouterr().out
    assert "from delegate_doctor import optimize" in text


def test_the_remaining_commands_still_dispatch(monkeypatch):
    from delegate_doctor import cli

    seen = {}
    monkeypatch.setattr(cli.android_setup, "setup_android_runners",
                        lambda **kwargs: seen.setdefault("setup", kwargs) and 0 or 0)
    assert cli.main(["setup-android"]) == 0
    assert "runners_dir" in seen["setup"]

    monkeypatch.setattr(cli, "run_optimize",
                        lambda target, **options: seen.setdefault(
                            "optimize", target) and 0 or 0)
    assert cli.main(["optimize", "m.py"]) == 0
    assert seen["optimize"] == "m.py"


# --- documentation stays in step --------------------------------------------

def test_the_readme_does_not_advertise_the_removed_command():
    text = (PROJECT_ROOT / "README.md").read_text()
    assert "delegate-doctor doctor" not in text


def test_the_readme_documents_every_runnable_example():
    """Each example is documented in the form it is actually usable in.

    Interface examples are runnable as a CLI command, so the README must show
    that exact command - a path that has moved is a path a judge cannot paste.
    """
    text = (PROJECT_ROOT / "README.md").read_text()
    assert "from delegate_doctor import optimize" in text
    for name in INTERFACE_EXAMPLES:
        assert f"delegate-doctor optimize examples/{name}" in text, \
            f"README omits examples/{name}"
    for name in PLAIN_SOURCE_EXAMPLES:
        assert f"examples/{name}" in text, f"README omits examples/{name}"
