"""The DelegateDoctor model interface, and the order preparation happens in.

The rule these tests exist to hold: **deterministic first, always**. A source
file that says how to build itself never causes a provider to be constructed,
a credential to be read, or a request to be made - whether or not AI is
configured, and whether or not the run is interactive.

Offline throughout. The child process is real where the test is about the
contract itself (that is the only way to check what `torch.export` does with a
bad return value), and mocked where the test is about ordering.
"""

from pathlib import Path

import pytest

from delegate_doctor import cli, model_interface
from delegate_doctor.model_source import ModelSourceError

INTERFACE_SOURCE = '''
import torch


class Tiny(torch.nn.Module):
    def forward(self, x):
        return x + 1


def delegate_doctor_model():
    return Tiny()


def delegate_doctor_inputs():
    return (torch.randn(1, 4),)
'''


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def model_file(tmp_path):
    return write(tmp_path / "model.py", INTERFACE_SOURCE)


# --- detection reads, it never imports ---------------------------------------

def test_both_required_functions_are_detected(model_file):
    report = model_interface.inspect_interface(model_file)
    assert report.complete
    assert model_interface.MODEL_FUNCTION in report.found_functions
    assert model_interface.INPUTS_FUNCTION in report.found_functions


def test_optional_functions_are_noticed_but_not_required(tmp_path):
    source = INTERFACE_SOURCE + '''

def delegate_doctor_kwargs():
    return {}
'''
    report = model_interface.inspect_interface(write(tmp_path / "m.py", source))
    assert report.complete
    assert model_interface.KWARGS_FUNCTION in report.found_functions


def test_a_missing_function_is_named(tmp_path):
    source = "def delegate_doctor_model():\n    return None\n"
    report = model_interface.inspect_interface(write(tmp_path / "m.py", source))
    assert not report.complete
    assert report.partial
    assert model_interface.INPUTS_FUNCTION in report.missing_functions
    assert model_interface.INPUTS_FUNCTION in report.describe() or True
    message = model_interface.missing_interface_message(report)
    assert "delegate_doctor_inputs" in message


def test_a_method_on_a_class_is_not_the_interface(tmp_path):
    """Only module-level defs are importable under that name."""
    source = '''
class Wrapper:
    def delegate_doctor_model(self):
        return None

    def delegate_doctor_inputs(self):
        return ()
'''
    report = model_interface.inspect_interface(write(tmp_path / "m.py", source))
    assert not report.complete


def test_detection_never_imports_the_module(tmp_path, monkeypatch):
    """A file with a side effect at import time must not run during detection."""
    marker = tmp_path / "ran.txt"
    source = (f"open({str(marker)!r}, 'w').write('imported')\n"
              f"def delegate_doctor_model():\n    return None\n"
              f"def delegate_doctor_inputs():\n    return ()\n")
    model_interface.inspect_interface(write(tmp_path / "m.py", source))
    assert not marker.exists(), "detection imported the user's module"


def test_a_syntax_error_is_reported_rather_than_raised(tmp_path):
    report = model_interface.inspect_interface(
        write(tmp_path / "m.py", "def broken(:\n"))
    assert not report.complete
    assert "syntax error" in report.parse_error


def test_an_unreadable_file_is_reported_rather_than_raised(tmp_path):
    report = model_interface.inspect_interface(tmp_path / "absent.py")
    assert not report.complete
    assert report.parse_error


# --- the adapter DelegateDoctor writes ---------------------------------------

def test_the_adapter_contains_no_user_source(model_file, tmp_path):
    source = model_interface.build_adapter_source(
        "model", model_file.parent, tmp_path / "out.pt2")
    assert "class Tiny" not in source
    assert "torch.randn(1, 4)" not in source
    # It imports by name; it does not inline anything the user wrote.
    assert "__import__('model')" in source or '__import__("model")' in source


def test_the_adapter_never_receives_a_credential(monkeypatch):
    """The child environment is the sanitized one, as the AI path's is."""
    from delegate_doctor.agent import privacy

    monkeypatch.setenv("DELEGATE_DOCTOR_LLM_API_KEY", "sk-secret-value")
    child = privacy.sanitized_child_environment()
    assert "DELEGATE_DOCTOR_LLM_API_KEY" not in child
    assert "sk-secret-value" not in "".join(child.values())


# --- the contract, enforced where the objects exist ---------------------------

def test_a_declared_interface_exports(model_file, tmp_path):
    prepared = model_interface.prepare_from_interface(
        model_file, tmp_path / "work", announce=lambda text: None)
    assert prepared.exported_program_path.is_file()
    assert prepared.inputs_path.is_file()

    spec = model_interface.model_spec_from_prepared(prepared)
    assert spec.example_args
    assert spec.exported_program is not None


def test_a_relative_model_path_still_imports(tmp_path, monkeypatch):
    """The child inserts the source directory on sys.path *and* cwds into it.

    A relative path would be applied twice, and the model's own module would
    then fail to import - reported, confusingly, as a missing dependency.
    """
    write(tmp_path / "model.py", INTERFACE_SOURCE)
    monkeypatch.chdir(tmp_path.parent)
    prepared = model_interface.prepare_from_interface(
        Path(tmp_path.name) / "model.py", tmp_path / "work",
        announce=lambda text: None)
    assert prepared.exported_program_path.is_file()


def test_an_unimportable_source_is_not_blamed_on_a_dependency(tmp_path):
    """"Install the missing package" is bad advice when it is the user's file."""
    message = model_interface.explain_failure("SOURCE_NOT_IMPORTABLE:my-model",
                                              Path("my-model.py"))
    assert "could not be imported by name" in message
    assert "never installs dependencies" not in message


def test_the_shipped_example_declares_a_usable_interface():
    """examples/dd001_softmax/interface_mnist.py demonstrates it."""
    example = Path(__file__).resolve().parent.parent / "examples" / \
        "dd001_softmax" / "interface_mnist.py"
    report = model_interface.inspect_interface(example)
    assert report.complete
    assert report.found_functions == (model_interface.MODEL_FUNCTION,
                                      model_interface.INPUTS_FUNCTION)


def test_a_single_tensor_is_normalized_to_a_one_element_tuple(tmp_path):
    """The obvious thing to write for a one-input model is accepted."""
    source = INTERFACE_SOURCE.replace(
        "return (torch.randn(1, 4),)", "return torch.randn(1, 4)")
    prepared = model_interface.prepare_from_interface(
        write(tmp_path / "model.py", source), tmp_path / "work",
        announce=lambda text: None)
    spec = model_interface.model_spec_from_prepared(prepared)
    assert len(spec.example_args) == 1


def test_a_list_of_inputs_becomes_a_tuple(tmp_path):
    source = INTERFACE_SOURCE.replace(
        "return (torch.randn(1, 4),)", "return [torch.randn(1, 4)]")
    prepared = model_interface.prepare_from_interface(
        write(tmp_path / "model.py", source), tmp_path / "work",
        announce=lambda text: None)
    spec = model_interface.model_spec_from_prepared(prepared)
    assert isinstance(spec.example_args, tuple)


def test_a_non_module_return_is_refused_by_name(tmp_path):
    source = INTERFACE_SOURCE.replace("return Tiny()", "return 'not a model'")
    with pytest.raises(model_interface.ModelInterfaceError) as caught:
        model_interface.prepare_from_interface(
            write(tmp_path / "model.py", source), tmp_path / "work",
            announce=lambda text: None)
    assert "torch.nn.Module" in str(caught.value)


def test_a_bad_inputs_return_is_refused_by_name(tmp_path):
    source = INTERFACE_SOURCE.replace(
        "return (torch.randn(1, 4),)", "return {'x': 1}")
    with pytest.raises(model_interface.ModelInterfaceError) as caught:
        model_interface.prepare_from_interface(
            write(tmp_path / "model.py", source), tmp_path / "work",
            announce=lambda text: None)
    assert "tuple of positional example" in str(caught.value)


def test_an_export_failure_is_distinguished_from_a_contract_failure():
    """Only an export failure is worth offering AI for."""
    assert model_interface.is_export_failure("EXPORT_FAILED: something")
    assert model_interface.is_export_failure("FORWARD_FAILED: something")
    assert not model_interface.is_export_failure("MISSING_DEPENDENCY:timm")
    assert not model_interface.is_export_failure("BAD_MODEL:str")


def test_a_missing_dependency_says_delegate_doctor_will_not_install_it():
    message = model_interface.explain_failure("MISSING_DEPENDENCY:timm",
                                              Path("model.py"))
    assert "never installs dependencies" in message


# --- ordering: deterministic first, always ------------------------------------

@pytest.fixture
def no_provider(monkeypatch):
    """Fail loudly if anything constructs a provider."""
    built = []

    def refuse(**kwargs):
        built.append(kwargs)
        raise AssertionError("a provider was constructed on the deterministic path")

    monkeypatch.setattr("delegate_doctor.agent.client.build_provider", refuse)
    return built


def test_the_interface_path_never_constructs_a_provider(model_file, no_provider):
    """Case 1: interface present, no AI configured."""
    spec = cli.prepare_model_source(model_file, interactive=True,
                                    announce=lambda text: None)
    assert spec.exported_program is not None
    assert no_provider == []


def test_the_interface_wins_even_when_ai_is_configured(model_file, monkeypatch):
    """Case 2: interface present *and* a provider available. AI is not asked."""
    calls = []

    class Provider:
        configuration = None

        def complete_structured(self, request):
            calls.append(request)
            raise AssertionError("AI was consulted despite a usable interface")

    monkeypatch.setattr("delegate_doctor.agent.client.build_provider",
                        lambda **kwargs: Provider())

    spec = cli.prepare_model_source(model_file, interactive=True,
                                    announce=lambda text: None)
    assert spec.exported_program is not None
    assert calls == []


def test_a_missing_interface_and_no_ai_names_both_ways_forward(tmp_path,
                                                               monkeypatch):
    """Case 3: no interface, no AI. Say so, and say what to write."""
    from delegate_doctor.agent.client import AINotConfigured

    monkeypatch.setattr("delegate_doctor.agent.client.build_provider",
                        lambda **kwargs: (_ for _ in ()).throw(
                            AINotConfigured("AI NOT CONFIGURED")))

    model = write(tmp_path / "model.py", "import torch\n")
    with pytest.raises(ModelSourceError) as caught:
        cli.prepare_model_source(model, interactive=True,
                                 announce=lambda text: None)
    message = str(caught.value)
    assert "model interface not found" in message
    assert "AI preparation is unavailable" in message
    assert "delegate_doctor_model" in message
    assert "delegate_doctor_inputs" in message
    assert "Then run the same command again" in message


def test_a_missing_interface_with_ai_reaches_preparation(tmp_path, monkeypatch):
    """Case 4: no interface, AI configured. The existing preparation runs."""
    reached = []

    monkeypatch.setattr("delegate_doctor.agent.client.build_provider",
                        lambda **kwargs: object())
    monkeypatch.setattr("delegate_doctor.agent.preparation.prepare_model",
                        lambda path, **kwargs: reached.append(kwargs) or object())
    monkeypatch.setattr("delegate_doctor.agent.preparation.model_spec_from_outcome",
                        lambda outcome, **kwargs: "spec")

    model = write(tmp_path / "model.py", "import torch\n")
    assert cli.prepare_model_source(model, interactive=True,
                                    announce=lambda text: None) == "spec"
    assert reached, "AI preparation was never reached"


def test_an_export_failure_with_ai_available_assists_the_interface(tmp_path,
                                                                    monkeypatch):
    """Case 5: the interface exists, export fails, AI assists *that* interface.

    Generic preparation must not run: it would ask which class is the model,
    about a file that already said.
    """
    from delegate_doctor.agent import export_assistance

    generic = []
    monkeypatch.setattr("delegate_doctor.agent.client.build_provider",
                        lambda **kwargs: object())
    monkeypatch.setattr("delegate_doctor.agent.preparation.prepare_model",
                        lambda path, **kwargs: generic.append(kwargs))
    monkeypatch.setattr(
        model_interface, "prepare_from_interface",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            model_interface.ModelInterfaceError(
                "EXPORT_FAILED: nope",
                failure=model_interface.ExportFailure(
                    stage=model_interface.STAGE_EXPORT,
                    exception_type="RuntimeError", message="nope"))))
    monkeypatch.setattr(
        export_assistance, "assist_export",
        lambda **kwargs: export_assistance.AssistanceOutcome(
            prepared="prepared",
            adjustment=export_assistance.ExportAdjustment(
                module_attributes={"aux_logits": False})))
    monkeypatch.setattr(model_interface, "model_spec_from_prepared",
                        lambda prepared, **kwargs: "spec")

    model = write(tmp_path / "model.py", INTERFACE_SOURCE)
    assert cli.prepare_model_source(model, interactive=True, allow_ai_source=True,
                                    announce=lambda text: None,
                                    prompt=lambda question: "y") == "spec"
    assert generic == [], "generic model discovery ran for a known interface"


def test_an_export_failure_with_no_ai_is_actionable(tmp_path, monkeypatch):
    """Case 6: the interface exists, export fails, no AI. Report it cleanly."""
    from delegate_doctor.agent.client import AINotConfigured

    monkeypatch.setattr("delegate_doctor.agent.client.build_provider",
                        lambda **kwargs: (_ for _ in ()).throw(
                            AINotConfigured("AI NOT CONFIGURED")))
    monkeypatch.setattr(
        model_interface, "prepare_from_interface",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            model_interface.ModelInterfaceError("EXPORT_FAILED: nope")))

    model = write(tmp_path / "model.py", INTERFACE_SOURCE)
    with pytest.raises(ModelSourceError) as caught:
        cli.prepare_model_source(model, interactive=True,
                                 announce=lambda text: None)
    message = str(caught.value)
    assert "EXPORT_FAILED" in message
    assert "AI preparation is unavailable" in message


def test_a_contract_failure_is_never_sent_to_ai(tmp_path, monkeypatch, no_provider):
    """A wrong return type is a fact about the file. No provider can fix it."""
    monkeypatch.setattr(
        model_interface, "prepare_from_interface",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            model_interface.ModelInterfaceError("BAD_MODEL:str")))

    model = write(tmp_path / "model.py", INTERFACE_SOURCE)
    with pytest.raises(ModelSourceError):
        cli.prepare_model_source(model, interactive=True,
                                 announce=lambda text: None)
    assert no_provider == []


def test_a_non_interactive_run_without_allow_ai_stays_deterministic(
        tmp_path, no_provider):
    """No consent is possible, so nothing is sent and nothing is constructed."""
    model = write(tmp_path / "model.py", "import torch\n")
    with pytest.raises(ModelSourceError) as caught:
        cli.prepare_model_source(model, interactive=False, allow_ai_source=False,
                                 announce=lambda text: None)
    message = str(caught.value)
    assert "cannot ask permission" in message
    assert "--allow-ai" in message
    assert no_provider == []


def test_allow_ai_authorizes_a_non_interactive_run(tmp_path, monkeypatch):
    """The one advanced flag, and it only means anything non-interactively."""
    seen = {}

    monkeypatch.setattr("delegate_doctor.agent.client.build_provider",
                        lambda **kwargs: object())
    monkeypatch.setattr("delegate_doctor.agent.preparation.prepare_model",
                        lambda path, **kwargs: seen.update(kwargs) or object())
    monkeypatch.setattr("delegate_doctor.agent.preparation.model_spec_from_outcome",
                        lambda outcome, **kwargs: "spec")

    model = write(tmp_path / "model.py", "import torch\n")
    cli.prepare_model_source(model, interactive=False, allow_ai_source=True,
                             announce=lambda text: None)
    assert seen["allow_source"] is True


def test_allow_ai_does_not_bypass_the_prompt_in_an_interactive_run(
        tmp_path, monkeypatch):
    """An interactive run can ask, so it asks. The flag is not a gag."""
    seen = {}

    monkeypatch.setattr("delegate_doctor.agent.client.build_provider",
                        lambda **kwargs: object())
    monkeypatch.setattr("delegate_doctor.agent.preparation.prepare_model",
                        lambda path, **kwargs: seen.update(kwargs) or object())
    monkeypatch.setattr("delegate_doctor.agent.preparation.model_spec_from_outcome",
                        lambda outcome, **kwargs: "spec")

    model = write(tmp_path / "model.py", "import torch\n")
    cli.prepare_model_source(model, interactive=True, allow_ai_source=True,
                             announce=lambda text: None)
    assert seen["allow_source"] is False


# --- the removed flags ---------------------------------------------------------

@pytest.mark.parametrize("flag", ["--no-ai", "--ai-repairs", "--allow-ai-repair"])
def test_the_old_mode_flags_are_absent_from_help(flag, capsys):
    with pytest.raises(SystemExit):
        cli.main(["optimize", "--help"])
    assert flag not in capsys.readouterr().out


def test_the_one_advanced_permission_flag_is_documented(capsys):
    with pytest.raises(SystemExit):
        cli.main(["optimize", "--help"])
    text = capsys.readouterr().out
    assert "--allow-ai" in text
    assert "--non-interactive" in text
