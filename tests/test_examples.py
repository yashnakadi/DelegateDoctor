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

# The demonstration workloads, each a standalone script.
DEMO_EXAMPLES = [
    "unet.py", "unetplusplus.py", "fpn.py", "pspnet.py",
    "deeplabv3plus.py", "linknet.py", "ghostnet.py", "mobilenet_v2.py",
]

# Architecture names that must not appear anywhere in the core package.
ARCHITECTURE_NAMES = (
    "unet", "Unet", "U-Net", "unetplusplus", "UnetPlusPlus",
    "pspnet", "PSPNet", "deeplabv3", "DeepLabV3", "linknet", "Linknet",
    "ghostnet", "GhostNet", "mobilenet", "MobileNet",
    "segmentation_models_pytorch", "timm", "torchvision",
)


def example_source(name: str) -> str:
    path = EXAMPLES_DIR / name
    assert path.is_file(), f"missing example: {name}"
    return path.read_text()


# --- every demo example exists and is a standalone script -------------------

@pytest.mark.parametrize("name", DEMO_EXAMPLES)
def test_the_example_exists_and_parses(name):
    ast.parse(example_source(name))


@pytest.mark.parametrize("name", DEMO_EXAMPLES)
def test_the_example_is_runnable_directly(name):
    """`python examples/unet.py` must actually do something."""
    tree = ast.parse(example_source(name))
    has_main = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        for node in tree.body
    )
    assert has_main, f"{name} has no `if __name__ == '__main__':` block"


@pytest.mark.parametrize("name", DEMO_EXAMPLES)
def test_the_example_imports_the_public_api(name):
    """`from delegate_doctor import optimize` - the same import a user writes."""
    tree = ast.parse(example_source(name))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "delegate_doctor"
        for alias in node.names
    }
    assert "optimize" in imported, f"{name} does not import the public optimize()"


@pytest.mark.parametrize("name", DEMO_EXAMPLES)
def test_the_example_calls_optimize_with_args(name):
    tree = ast.parse(example_source(name))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "optimize"
    ]
    assert calls, f"{name} never calls optimize()"
    assert any(kw.arg == "args" for call in calls for kw in call.keywords), \
        f"{name} does not pass args= to optimize()"


@pytest.mark.parametrize("name", DEMO_EXAMPLES)
def test_the_example_builds_a_model(name):
    tree = ast.parse(example_source(name))
    functions = {node.name for node in tree.body
                 if isinstance(node, ast.FunctionDef)}
    assert "build_model" in functions, f"{name} has no build_model()"


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
    """Examples construct models. They never mention DD rules in code."""
    source = example_source(name)
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree) or ""
    code = source.replace(docstring, "")
    for token in ("DD-001", "DD-002", "softmax(dim", "aten.alias", "alias"):
        assert token not in code, f"{name} contains rule-specific code: {token}"


@pytest.mark.parametrize("name", DEMO_EXAMPLES)
def test_the_example_opens_the_report_at_the_end(name):
    """The demo experience: run the script, get the report in a browser."""
    tree = ast.parse(example_source(name))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "open_report"
    ]
    assert calls, f"{name} never calls result.open_report()"


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
                        lambda target, inputs, **options: seen.setdefault(
                            "optimize", (target, inputs)) and 0 or 0)
    assert cli.main(["optimize", "m.pt2", "--inputs", "i.pt"]) == 0
    assert seen["optimize"] == ("m.pt2", "i.pt")


# --- documentation stays in step --------------------------------------------

def test_the_readme_does_not_advertise_the_removed_command():
    text = (PROJECT_ROOT / "README.md").read_text()
    assert "delegate-doctor doctor" not in text


def test_the_readme_shows_the_python_api_and_the_examples():
    text = (PROJECT_ROOT / "README.md").read_text()
    assert "from delegate_doctor import optimize" in text
    for name in DEMO_EXAMPLES:
        assert f"python examples/{name}" in text, f"README omits examples/{name}"
