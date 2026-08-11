"""Tests for `delegate-doctor optimize` - loading a user's own model file.

Fully offline: tiny synthetic models, pytest tmp_path, no adb / device / NDK /
network / native runner.
"""

import sys
import textwrap

import pytest
import torch

from delegate_doctor import custom_model
from delegate_doctor.custom_model import CustomModelError
from delegate_doctor.export_model import ModelSpec, export_to_aten
from delegate_doctor.repairs import dd001_softmax, dd002_noop_alias

VALID = """
    import torch

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 1)

        def forward(self, x):
            return self.conv(x)

    def create_model():
        model = Tiny()
        model.eval()
        return model

    def example_inputs():
        return (torch.randn(1, 3, 8, 8),)
"""


def write(tmp_path, source, name="my_model.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(source))
    return str(path)


# --- happy path ------------------------------------------------------------

def test_valid_custom_module_loads(tmp_path):
    spec = custom_model.load(write(tmp_path, VALID))

    assert isinstance(spec, ModelSpec)
    assert isinstance(spec.model, torch.nn.Module)
    assert isinstance(spec.example_inputs, tuple)
    assert tuple(spec.example_inputs[0].shape) == (1, 3, 8, 8)
    # Unknown output semantics, so no argmax claim is made for custom models.
    assert spec.argmax_dim is None


def test_absolute_and_relative_paths_both_work(tmp_path, monkeypatch):
    absolute = write(tmp_path, VALID)
    assert custom_model.load(absolute).name == "my_model.py"

    monkeypatch.chdir(tmp_path)
    assert custom_model.load("my_model.py").name == "my_model.py"
    assert custom_model.load("./my_model.py").name == "my_model.py"


def test_display_name_comes_from_the_filename(tmp_path):
    spec = custom_model.load(write(tmp_path, VALID, "my_medical_segmenter.py"))
    assert spec.name == "my_medical_segmenter.py"


def test_module_level_model_name_wins_when_present(tmp_path):
    spec = custom_model.load(
        write(tmp_path, VALID + '\n    MODEL_NAME = "Medical Segmenter"\n')
    )
    assert spec.name == "Medical Segmenter"


def test_multiple_positional_inputs_are_supported(tmp_path):
    source = """
        import torch

        class TwoInput(torch.nn.Module):
            def forward(self, a, b):
                return a + b

        def create_model():
            return TwoInput().eval()

        def example_inputs():
            return (torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4))
    """
    spec = custom_model.load(write(tmp_path, source))
    assert len(spec.example_inputs) == 2
    assert "2 tensors" in custom_model.describe_inputs(spec.example_inputs)


# --- file-level errors -----------------------------------------------------

def test_missing_file(tmp_path):
    with pytest.raises(CustomModelError) as caught:
        custom_model.load(str(tmp_path / "absent.py"))
    assert "not found" in str(caught.value)


def test_non_python_file(tmp_path):
    path = tmp_path / "model.txt"
    path.write_text("not python")
    with pytest.raises(CustomModelError) as caught:
        custom_model.load(str(path))
    assert "Not a Python file" in str(caught.value)


def test_directory_is_rejected(tmp_path):
    with pytest.raises(CustomModelError):
        custom_model.load(str(tmp_path))


def test_import_error_is_reported_without_a_traceback(tmp_path):
    with pytest.raises(CustomModelError) as caught:
        custom_model.load(write(tmp_path, "import a_module_that_does_not_exist\n"))
    assert "Failed to import" in str(caught.value)
    assert "ModuleNotFoundError" in str(caught.value)


# --- contract errors -------------------------------------------------------

def test_missing_create_model(tmp_path):
    source = "import torch\ndef example_inputs():\n    return (torch.randn(1),)\n"
    with pytest.raises(CustomModelError) as caught:
        custom_model.load(write(tmp_path, source))
    assert "create_model()" in str(caught.value)


def test_missing_example_inputs(tmp_path):
    source = "import torch\ndef create_model():\n    return torch.nn.Identity()\n"
    with pytest.raises(CustomModelError) as caught:
        custom_model.load(write(tmp_path, source))
    assert "example_inputs()" in str(caught.value)


def test_non_callable_create_model(tmp_path):
    source = "import torch\ncreate_model = 5\ndef example_inputs():\n    return (torch.randn(1),)\n"
    with pytest.raises(CustomModelError) as caught:
        custom_model.load(write(tmp_path, source))
    assert "not callable" in str(caught.value)


def test_non_callable_example_inputs(tmp_path):
    source = "import torch\ndef create_model():\n    return torch.nn.Identity()\nexample_inputs = 5\n"
    with pytest.raises(CustomModelError) as caught:
        custom_model.load(write(tmp_path, source))
    assert "not callable" in str(caught.value)


def test_create_model_returning_non_module(tmp_path):
    source = "import torch\ndef create_model():\n    return 'nope'\ndef example_inputs():\n    return (torch.randn(1),)\n"
    with pytest.raises(CustomModelError) as caught:
        custom_model.load(write(tmp_path, source))
    assert "torch.nn.Module" in str(caught.value)


def test_example_inputs_returning_a_bare_tensor_suggests_the_fix(tmp_path):
    source = "import torch\ndef create_model():\n    return torch.nn.Identity()\ndef example_inputs():\n    return torch.randn(1, 3)\n"
    with pytest.raises(CustomModelError) as caught:
        custom_model.load(write(tmp_path, source))
    message = str(caught.value)
    assert "tuple[torch.Tensor, ...]" in message
    assert "return (tensor,)" in message


def test_empty_input_tuple(tmp_path):
    source = "import torch\ndef create_model():\n    return torch.nn.Identity()\ndef example_inputs():\n    return ()\n"
    with pytest.raises(CustomModelError) as caught:
        custom_model.load(write(tmp_path, source))
    assert "empty" in str(caught.value)


def test_unsupported_input_type(tmp_path):
    source = "import torch\ndef create_model():\n    return torch.nn.Identity()\ndef example_inputs():\n    return (torch.randn(1, 3), 'text')\n"
    with pytest.raises(CustomModelError) as caught:
        custom_model.load(write(tmp_path, source))
    assert "expected torch.Tensor" in str(caught.value)


def test_unsupported_dtype_is_rejected_not_converted(tmp_path):
    source = "import torch\ndef create_model():\n    return torch.nn.Identity()\ndef example_inputs():\n    return (torch.randn(1, 3).half(),)\n"
    with pytest.raises(CustomModelError) as caught:
        custom_model.load(write(tmp_path, source))
    message = str(caught.value)
    assert "torch.float16" in message
    assert "fp32" in message


def test_create_model_exception_is_surfaced(tmp_path):
    source = "def create_model():\n    raise ValueError('bad checkpoint')\ndef example_inputs():\n    return ()\n"
    with pytest.raises(CustomModelError) as caught:
        custom_model.load(write(tmp_path, source))
    assert "bad checkpoint" in str(caught.value)


# --- output contract -------------------------------------------------------

def test_unsupported_output_structure_is_rejected(tmp_path):
    """Device verification reads back one fp32 tensor, so a dict cannot work."""
    source = """
        import torch

        class DictOut(torch.nn.Module):
            def forward(self, x):
                return {"logits": x}

        def create_model():
            return DictOut().eval()

        def example_inputs():
            return (torch.randn(1, 3),)
    """
    with pytest.raises(CustomModelError) as caught:
        custom_model.load(write(tmp_path, source))
    assert "Unsupported model output" in str(caught.value)


def test_tuple_output_is_accepted_via_its_first_tensor(tmp_path):
    source = """
        import torch

        class TupleOut(torch.nn.Module):
            def forward(self, x):
                return (x * 2, x * 3)

        def create_model():
            return TupleOut().eval()

        def example_inputs():
            return (torch.randn(1, 3),)
    """
    assert custom_model.load(write(tmp_path, source)) is not None


# --- behaviour guarantees --------------------------------------------------

def test_example_inputs_is_called_exactly_once(tmp_path):
    """Random inputs must be frozen: every stage has to see the same values."""
    source = """
        import torch

        CALLS = []

        class Tiny(torch.nn.Module):
            def forward(self, x):
                return x * 2

        def create_model():
            return Tiny().eval()

        def example_inputs():
            CALLS.append(1)
            return (torch.randn(1, 3, 4, 4),)
    """
    spec = custom_model.load(write(tmp_path, source))
    first = spec.example_inputs[0].clone()
    # the spec keeps the same tensor objects, not a fresh draw
    assert torch.equal(spec.example_inputs[0], first)
    assert spec.example_inputs[0] is spec.example_inputs[0]


def test_training_mode_model_is_switched_to_eval(tmp_path):
    source = """
        import torch

        def create_model():
            return torch.nn.Sequential(torch.nn.Dropout(0.5))   # left in train mode

        def example_inputs():
            return (torch.randn(1, 3),)
    """
    spec = custom_model.load(write(tmp_path, source))
    assert spec.model.training is False


def test_loaded_modules_get_unique_names(tmp_path):
    """The module must stay in sys.modules - torch.export re-imports it while
    tracing - so the generated name has to be unique per load."""
    before = {n for n in sys.modules if "delegate_doctor_custom" in n}
    custom_model.load(write(tmp_path, VALID, "one.py"))
    custom_model.load(write(tmp_path, VALID, "two.py"))
    added = {n for n in sys.modules if "delegate_doctor_custom" in n} - before
    assert len(added) == 2, "each load needs its own module name"


def test_repeated_loads_do_not_collide(tmp_path):
    a = custom_model.load(write(tmp_path, VALID, "a.py"))
    b = custom_model.load(write(tmp_path, VALID, "b.py"))
    assert a.name == "a.py" and b.name == "b.py"


# --- the catalog applies automatically -------------------------------------

def test_custom_model_can_match_dd001(tmp_path):
    """No --repair flag: the rule is found from graph semantics."""
    source = """
        import torch

        class Seg(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 1)

            def forward(self, x):
                return torch.softmax(self.conv(x), dim=1)

        def create_model():
            return Seg().eval()

        def example_inputs():
            return (torch.randn(1, 3, 8, 8),)
    """
    spec = custom_model.load(write(tmp_path, source))
    exported = export_to_aten(spec.model, spec.example_inputs)
    assert dd001_softmax.detect(exported).applies
    assert not dd002_noop_alias.detect(exported).applies


def test_custom_model_can_match_dd002(tmp_path):
    source = """
        import torch

        class Aliased(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, 1)

            def forward(self, x):
                return self.conv(x)[:, :4, :, :]      # full-width slice -> alias

        def create_model():
            return Aliased().eval()

        def example_inputs():
            return (torch.randn(1, 3, 8, 8),)
    """
    spec = custom_model.load(write(tmp_path, source))
    exported = export_to_aten(spec.model, spec.example_inputs)
    assert dd002_noop_alias.detect(exported).applies


def test_custom_model_with_no_matching_repair_is_not_an_error(tmp_path):
    """A model the catalog cannot help is a valid, successful analysis."""
    spec = custom_model.load(write(tmp_path, VALID))
    exported = export_to_aten(spec.model, spec.example_inputs)
    assert not dd001_softmax.detect(exported).applies
    assert not dd002_noop_alias.detect(exported).applies


# --- CLI plumbing ----------------------------------------------------------

def test_cli_dispatches_optimize(tmp_path, monkeypatch):
    from delegate_doctor import cli

    calls = []
    monkeypatch.setattr(cli, "run_optimize",
                        lambda model_file, **kw: calls.append((model_file, kw)) or 0)
    path = write(tmp_path, VALID)
    assert cli.main(["optimize", path]) == 0
    assert calls[0][0] == path
    assert calls[0][1]["threads"] == 4


def test_cli_reports_contract_errors_without_a_traceback(tmp_path, capsys):
    from delegate_doctor import cli

    source = "import torch\ndef create_model():\n    return torch.nn.Identity()\n"
    assert cli.main(["optimize", write(tmp_path, source)]) == 2
    assert "example_inputs()" in capsys.readouterr().err


def test_builtin_doctor_command_is_unchanged(monkeypatch):
    """`doctor` must keep working exactly as before."""
    from delegate_doctor import cli

    seen = {}
    monkeypatch.setattr(cli, "run_doctor",
                        lambda model, seed, **kw: seen.update(model=model, seed=seed) or 0)
    assert cli.main(["doctor", "pspnet"]) == 0
    assert seen["model"] == "pspnet"


def test_both_commands_share_one_pipeline():
    """doctor and optimize must converge on run_optimization, not fork."""
    import inspect

    from delegate_doctor import cli

    assert "run_optimization" in inspect.getsource(cli.run_doctor)
    assert "run_optimization" in inspect.getsource(cli.run_optimize)
