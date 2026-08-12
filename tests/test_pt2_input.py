"""The `.pt2` + `inputs.pt` boundary: what is accepted, and what is refused early.

Fully offline. Models here are two-layer toys so exporting them costs
milliseconds; nothing touches adb, a device, the network or a runner binary.
"""

import os

import pytest
import torch

from delegate_doctor import pt2_input
from delegate_doctor.export_model import ModelSpec
from delegate_doctor.pt2_input import ModelInputError


# --- fixtures ---------------------------------------------------------------

class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, x):
        return torch.softmax(self.conv(x), dim=1)


INPUT_SHAPE = (1, 3, 8, 8)


class CustomPayload:
    """An arbitrary object in an inputs file. Module level so it can be pickled
    at all - the point of the test is that *loading* refuses it."""

    def __init__(self):
        self.data = [1, 2, 3]


def build_reference():
    """The exact construction order the `exported` fixture uses, so weights match."""
    torch.manual_seed(0)
    inputs = (torch.randn(*INPUT_SHAPE),)
    return TinyNet().eval(), inputs


@pytest.fixture
def exported(tmp_path):
    """A real model.pt2 + inputs.pt pair on disk."""
    model, inputs = build_reference()
    program = torch.export.export(model, inputs)

    model_path = str(tmp_path / "model.pt2")
    inputs_path = str(tmp_path / "inputs.pt")
    torch.export.save(program, model_path)
    torch.save(inputs, inputs_path)
    return model_path, inputs_path


# --- model path validation --------------------------------------------------

@pytest.mark.parametrize("target", [
    "https://github.com/owner/repo",
    "http://example.com/model.pt2",
    "github.com/owner/repo",
    "git@github.com:owner/repo.git",
    "https://huggingface.co/some/model",
])
def test_urls_are_rejected(target):
    with pytest.raises(ModelInputError) as caught:
        pt2_input.resolve_model_path(target)
    assert "unsupported model input" in str(caught.value)
    assert ".pt2" in str(caught.value)


def test_directory_is_rejected(tmp_path):
    with pytest.raises(ModelInputError) as caught:
        pt2_input.resolve_model_path(str(tmp_path))
    assert "directory" in str(caught.value)


def test_missing_model_file_is_rejected(tmp_path):
    with pytest.raises(ModelInputError) as caught:
        pt2_input.resolve_model_path(str(tmp_path / "nope.pt2"))
    assert "not found" in str(caught.value)


def test_a_pte_is_not_mistaken_for_a_pt2(tmp_path):
    """The two extensions are easy to confuse and mean opposite things."""
    path = tmp_path / "model.pte"
    path.write_bytes(b"\x00")
    with pytest.raises(ModelInputError) as caught:
        pt2_input.resolve_model_path(str(path))
    message = str(caught.value)
    assert ".pt2 = PyTorch ExportedProgram" in message
    assert ".pte = ExecuTorch deployment artifact" in message


def test_python_source_is_no_longer_a_model_input(tmp_path):
    path = tmp_path / "model.py"
    path.write_text("import torch\n")
    with pytest.raises(ModelInputError) as caught:
        pt2_input.resolve_model_path(str(path))
    assert "no longer a DelegateDoctor input" in str(caught.value)
    assert "torch.export.save" in str(caught.value)


def test_wrong_suffix_is_reported_clearly(tmp_path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"\x00")
    with pytest.raises(ModelInputError) as caught:
        pt2_input.resolve_model_path(str(path))
    assert "must be a .pt2 file" in str(caught.value)


def test_special_files_are_rejected(tmp_path):
    fifo = tmp_path / "pipe.pt2"
    try:
        os.mkfifo(fifo)
    except (AttributeError, OSError):
        pytest.skip("this platform cannot create a FIFO")
    with pytest.raises(ModelInputError) as caught:
        pt2_input.resolve_model_path(str(fifo))
    assert "not a regular file" in str(caught.value)


# --- loading the exported program -------------------------------------------

def test_a_valid_pt2_loads_as_an_exported_program(exported):
    model_path, _ = exported
    program = pt2_input.load_exported_program(model_path)
    assert isinstance(program, torch.export.ExportedProgram)
    assert callable(program.module())


def test_a_corrupt_pt2_fails_clearly(tmp_path):
    path = tmp_path / "broken.pt2"
    path.write_bytes(b"this is not an exported program")
    with pytest.raises(ModelInputError) as caught:
        pt2_input.load_exported_program(str(path))
    assert "Could not load the exported program" in str(caught.value)


def test_a_pt2_holding_something_else_is_rejected(tmp_path):
    """A file with the right name but the wrong contents."""
    path = tmp_path / "notaprogram.pt2"
    torch.save({"weights": torch.randn(2)}, path)
    with pytest.raises(ModelInputError) as caught:
        pt2_input.load_exported_program(str(path))
    # Either the loader refuses it or we catch the wrong type; both are clear.
    message = str(caught.value)
    assert ("did not contain an ExportedProgram" in message
            or "Could not load the exported program" in message)


# --- loading the inputs -----------------------------------------------------

def test_a_tensor_tuple_loads(exported):
    _, inputs_path = exported
    inputs = pt2_input.load_inputs(inputs_path)
    assert isinstance(inputs, tuple)
    assert tuple(inputs[0].shape) == INPUT_SHAPE


def test_a_bare_tensor_is_wrapped(tmp_path):
    path = tmp_path / "inputs.pt"
    torch.save(torch.randn(*INPUT_SHAPE), path)
    inputs = pt2_input.load_inputs(str(path))
    assert isinstance(inputs, tuple) and len(inputs) == 1


def test_a_list_is_accepted(tmp_path):
    path = tmp_path / "inputs.pt"
    torch.save([torch.randn(*INPUT_SHAPE)], path)
    assert isinstance(pt2_input.load_inputs(str(path)), tuple)


def test_missing_inputs_file_is_rejected(tmp_path):
    with pytest.raises(ModelInputError) as caught:
        pt2_input.resolve_inputs_path(str(tmp_path / "nope.pt"))
    assert "Inputs file not found" in str(caught.value)


def test_a_dict_of_kwargs_is_unsupported(tmp_path):
    path = tmp_path / "inputs.pt"
    torch.save({"x": torch.randn(*INPUT_SHAPE)}, path)
    with pytest.raises(ModelInputError) as caught:
        pt2_input.load_inputs(str(path))
    assert "Unsupported input structure" in str(caught.value)
    assert "Keyword arguments" in str(caught.value)


def test_a_non_tensor_element_is_unsupported(tmp_path):
    path = tmp_path / "inputs.pt"
    torch.save((torch.randn(*INPUT_SHAPE), 5), path)
    with pytest.raises(ModelInputError) as caught:
        pt2_input.load_inputs(str(path))
    assert "Input 1 is int" in str(caught.value)


def test_an_empty_input_tuple_is_rejected(tmp_path):
    path = tmp_path / "inputs.pt"
    torch.save((), path)
    with pytest.raises(ModelInputError) as caught:
        pt2_input.load_inputs(str(path))
    assert "no inputs" in str(caught.value)


def test_non_fp32_inputs_are_rejected(tmp_path):
    path = tmp_path / "inputs.pt"
    torch.save((torch.randint(0, 5, INPUT_SHAPE),), path)
    with pytest.raises(ModelInputError) as caught:
        pt2_input.load_inputs(str(path))
    assert "Unsupported input dtype" in str(caught.value)


def test_non_finite_inputs_are_rejected(tmp_path):
    path = tmp_path / "inputs.pt"
    tensor = torch.randn(*INPUT_SHAPE)
    tensor[0, 0, 0, 0] = float("nan")
    torch.save((tensor,), path)
    with pytest.raises(ModelInputError) as caught:
        pt2_input.load_inputs(str(path))
    assert "NaN or infinity" in str(caught.value)


# --- input deserialization safety -------------------------------------------

def test_inputs_are_loaded_with_weights_only(monkeypatch, exported):
    """The restricted unpickler is not optional."""
    _, inputs_path = exported
    seen = {}
    real_load = torch.load

    def spy(path, **kwargs):
        seen.update(kwargs)
        return real_load(path, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    pt2_input.load_inputs(inputs_path)
    assert seen.get("weights_only") is True


def test_an_input_artifact_holding_a_custom_object_is_rejected(tmp_path):
    """weights_only=True must refuse it rather than unpickling it."""
    path = tmp_path / "inputs.pt"
    torch.save((CustomPayload(),), path)

    with pytest.raises(ModelInputError) as caught:
        pt2_input.load_inputs(str(path))
    message = str(caught.value)
    assert "Could not load the inputs file" in message
    assert "weights_only=True" in message


def test_there_is_no_unrestricted_load_fallback():
    """A convenience fallback would defeat the point; assert it never appears."""
    import ast

    tree = ast.parse(open(pt2_input.__file__).read())
    loads = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "load"
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"
    ]
    assert len(loads) == 1, "there must be exactly one torch.load call"
    flags = {kw.arg: kw.value.value for kw in loads[0].keywords
             if isinstance(kw.value, ast.Constant)}
    assert flags.get("weights_only") is True


# --- execution against the exported program ---------------------------------

def test_baseline_execution_succeeds(exported):
    model_path, inputs_path = exported
    program = pt2_input.load_exported_program(model_path)
    inputs = pt2_input.load_inputs(inputs_path)
    output = pt2_input.check_executes(program, inputs, "model.pt2")
    assert isinstance(output, torch.Tensor)
    assert output.dtype == torch.float32


def test_incompatible_inputs_fail_before_optimization(tmp_path, exported):
    """A shape the graph cannot take must be caught at load, not on the device."""
    model_path, _ = exported
    wrong = tmp_path / "wrong.pt"
    torch.save((torch.randn(1, 3, 64, 64),), wrong)

    with pytest.raises(ModelInputError) as caught:
        pt2_input.load_model_spec(model_path, str(wrong))
    message = str(caught.value)
    assert "not compatible with the exported program" in message
    assert "[1, 3, 64, 64]" in message


def test_wrong_number_of_inputs_fails(tmp_path, exported):
    model_path, _ = exported
    wrong = tmp_path / "wrong.pt"
    torch.save((torch.randn(*INPUT_SHAPE), torch.randn(*INPUT_SHAPE)), wrong)
    with pytest.raises(ModelInputError):
        pt2_input.load_model_spec(model_path, str(wrong))


# --- the assembled ModelSpec ------------------------------------------------

def test_load_model_spec_produces_an_exported_program_spec(exported):
    model_path, inputs_path = exported
    spec = pt2_input.load_model_spec(model_path, inputs_path)

    assert isinstance(spec, ModelSpec)
    assert isinstance(spec.exported_program, torch.export.ExportedProgram)
    assert spec.name == "model.pt2"
    assert tuple(spec.example_args[0].shape) == INPUT_SHAPE
    # Output semantics of a user's graph are unknown, so no argmax claim.
    assert spec.argmax_dim is None


def test_the_model_spec_no_longer_carries_an_nn_module(exported):
    """The graph is the unit of work now."""
    model_path, inputs_path = exported
    spec = pt2_input.load_model_spec(model_path, inputs_path)
    assert not hasattr(spec, "model")


def test_the_loaded_program_matches_the_original_model(exported):
    """Round-tripping through .pt2 must not change what the model computes."""
    model_path, inputs_path = exported
    spec = pt2_input.load_model_spec(model_path, inputs_path)

    reference, _ = build_reference()
    with torch.no_grad():
        expected = reference(*spec.example_args)
        actual = spec.exported_program.module()(*spec.example_args)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_describe_inputs_summarizes_without_printing_values():
    text = pt2_input.describe_inputs((torch.randn(1, 3, 8, 8),))
    assert text == "1 tensor · fp32 · [1,3,8,8]"
