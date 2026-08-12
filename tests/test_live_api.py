"""The public Python API: `from delegate_doctor import optimize`.

Offline. No device is attached during tests, which is itself one of the things
being tested - a missing target is a capability limit, not a failure, and the
run still returns a real analysis.
"""

import copy

import pytest
import torch

import delegate_doctor
from delegate_doctor import (
    ExportFailed,
    OptimizationResult,
    analyze_exported_program,
    optimize,
)
from delegate_doctor import api, result as result_module


# --- fixtures ---------------------------------------------------------------

class SoftmaxNet(torch.nn.Module):
    """Carries the DD-001 pattern: softmax on a non-last dimension."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return torch.softmax(x * self.scale, dim=1)


class TwoInputNet(torch.nn.Module):
    def forward(self, x, mask):
        return x * mask


class KwargNet(torch.nn.Module):
    def forward(self, x, attention_mask=None):
        if attention_mask is not None:
            x = x * attention_mask
        return x + 1


class IntInputNet(torch.nn.Module):
    def forward(self, ids):
        return ids.float() + 1.0


class TwoOutputNet(torch.nn.Module):
    def forward(self, x):
        return x + 1, x * 2


class UnexportableNet(torch.nn.Module):
    def forward(self, x):
        # A data-dependent branch torch.export cannot capture.
        if x.sum() > 0:
            return x + 1
        raise RuntimeError("negative")


def run(model, **kwargs):
    """optimize() into a scratch artifacts dir, quietly."""
    return optimize(model, quiet=True, **kwargs)


@pytest.fixture(autouse=True)
def artifacts(tmp_path, monkeypatch):
    """Keep every run's artifacts inside the test's own tmp_path."""
    monkeypatch.setattr(api.pipeline, "DEFAULT_ARTIFACTS_DIR", str(tmp_path / "art"))
    original = api.pipeline.run_optimization

    def with_tmp(spec, **options):
        options.setdefault("artifacts_dir", str(tmp_path / "art"))
        return original(spec, **options)

    monkeypatch.setattr(api.pipeline, "run_optimization", with_tmp)


# --- acceptance criterion A -------------------------------------------------

def test_the_public_import_works():
    assert callable(delegate_doctor.optimize)
    assert "optimize" in delegate_doctor.__all__


def test_a_simple_model_is_analyzed(tmp_path):
    outcome = run(SoftmaxNet(), args=(torch.randn(1, 4, 8, 8),))
    assert isinstance(outcome, OptimizationResult)
    assert outcome.stage_status(result_module.EXPORT) == result_module.PASS
    assert outcome.stage_status(result_module.LOWERING) == result_module.PASS
    assert outcome.stage_status(result_module.DELEGATION) == result_module.PASS
    assert outcome.analyzed


def test_the_result_exposes_the_documented_fields():
    outcome = run(SoftmaxNet(), args=(torch.randn(1, 4, 8, 8),))
    assert isinstance(outcome.status, str)
    assert isinstance(outcome.repair_available, bool)
    assert outcome.output_pte is None      # no device, so nothing was published
    assert outcome.to_dict()["status"] == outcome.status


# --- the caller's model is not damaged --------------------------------------

def test_training_mode_is_restored():
    model = SoftmaxNet()
    model.train()
    run(model, args=(torch.randn(1, 4, 8, 8),))
    assert model.training, "optimize() left the caller's model in eval mode"


def test_eval_mode_stays_eval():
    model = SoftmaxNet().eval()
    run(model, args=(torch.randn(1, 4, 8, 8),))
    assert not model.training


def test_parameters_are_not_modified():
    model = SoftmaxNet()
    with torch.no_grad():
        model.scale.fill_(3.5)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    run(model, args=(torch.randn(1, 4, 8, 8),))

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name]), f"{name} was modified"


def test_the_caller_keeps_their_own_object():
    """No copy is substituted; the trained weights that were passed are used."""
    model = SoftmaxNet()
    with torch.no_grad():
        model.scale.fill_(2.0)

    exported = api.export_for_analysis(model, args=(torch.randn(1, 4, 8, 8),))
    values = [t for t in exported.state_dict.values()]
    assert any(torch.allclose(t, torch.tensor([2.0])) for t in values), \
        "the exported graph does not carry the supplied weights"


def test_a_dataparallel_wrapper_is_read_through_not_unwrapped():
    model = SoftmaxNet()
    wrapped = torch.nn.DataParallel(model)
    api.export_for_analysis(wrapped, args=(torch.randn(1, 4, 8, 8),))
    assert isinstance(wrapped, torch.nn.DataParallel)
    assert wrapped.module is model


def test_a_non_cpu_model_is_refused_rather_than_moved(monkeypatch):
    """DelegateDoctor will not call .cpu() on someone's live model."""
    model = SoftmaxNet()

    class FakeDevice:
        type = "cuda"

    monkeypatch.setattr(type(model.scale), "device",
                        property(lambda self: FakeDevice()), raising=False)
    with pytest.raises(ExportFailed) as caught:
        api.export_for_analysis(model, args=(torch.randn(1, 4, 8, 8),))
    assert "will not move your model" in str(caught.value)


# --- export is the acceptance boundary --------------------------------------

def test_export_failure_is_framed_clearly():
    with pytest.raises(ExportFailed) as caught:
        api.export_for_analysis(UnexportableNet(), args=(torch.randn(4),))
    message = str(caught.value)
    assert "PYTORCH EXPORT FAILED" in message
    assert "has not entered the DelegateDoctor analysis pipeline" in message
    assert "PyTorch error:" in message


def test_a_non_module_is_rejected():
    with pytest.raises(ExportFailed) as caught:
        api.export_for_analysis("not a model", args=())
    assert "takes a torch.nn.Module" in str(caught.value)


def test_multiple_positional_args_are_forwarded():
    outcome = run(TwoInputNet(), args=(torch.randn(1, 4), torch.randn(1, 4)))
    assert outcome.stage_status(result_module.EXPORT) == result_module.PASS


def test_kwargs_are_forwarded_to_torch_export():
    outcome = run(KwargNet(),
                  args=(torch.randn(1, 4),),
                  kwargs={"attention_mask": torch.randn(1, 4)})
    assert outcome.stage_status(result_module.EXPORT) == result_module.PASS


def test_dynamic_shapes_reach_torch_export(monkeypatch):
    seen = {}
    real_export = torch.export.export

    def spy(model, args=(), kwargs=None, **options):
        seen.update(options)
        return real_export(model, args=args, kwargs=kwargs, **options)

    monkeypatch.setattr(torch.export, "export", spy)
    batch = torch.export.Dim("batch", min=1, max=8)
    api.export_for_analysis(SoftmaxNet(), args=(torch.randn(2, 4, 8, 8),),
                            dynamic_shapes=({0: batch},))
    assert seen["dynamic_shapes"] == ({0: batch},)


def test_a_dynamic_graph_is_analyzed_even_when_a_rule_declines_it():
    """Recognising a pattern and being willing to rewrite it are different."""
    batch = torch.export.Dim("batch", min=1, max=8)
    outcome = run(SoftmaxNet(), args=(torch.randn(2, 4, 8, 8),),
                  dynamic_shapes=({0: batch},))
    assert outcome.stage_status(result_module.EXPORT) == result_module.PASS
    assert outcome.stage_status(result_module.LOWERING) == result_module.PASS
    # DD-001 declines dynamic shapes; the model is analyzed regardless.
    assert outcome.analyzed


# --- dtypes and outputs are not initial rejections --------------------------

def test_an_int64_input_is_exported_and_analyzed():
    outcome = run(IntInputNet(), args=(torch.randint(0, 5, (1, 4)),))
    assert outcome.stage_status(result_module.EXPORT) == result_module.PASS
    assert outcome.analyzed


def test_an_int64_input_stops_at_the_device_transport_not_at_export():
    outcome = run(IntInputNet(), args=(torch.randint(0, 5, (1, 4)),))
    device_stage = outcome.stage(result_module.DEVICE)
    assert device_stage.status in (result_module.UNSUPPORTED,
                                   result_module.UNAVAILABLE)
    # Nothing may claim a measurement that never happened.
    assert outcome.before_profile is None
    assert outcome.benchmark is None
    assert outcome.stage_status(result_module.BENCHMARK) == result_module.NOT_RUN


def test_multiple_outputs_do_not_block_export():
    outcome = run(TwoOutputNet(), args=(torch.randn(1, 4),))
    assert outcome.stage_status(result_module.EXPORT) == result_module.PASS
    assert outcome.analyzed


# --- the two entry points converge ------------------------------------------

def test_analyze_exported_program_accepts_a_ready_graph():
    program = torch.export.export(SoftmaxNet().eval(), (torch.randn(1, 4, 8, 8),))
    outcome = analyze_exported_program(program, args=(torch.randn(1, 4, 8, 8),),
                                       quiet=True)
    assert outcome.stage_status(result_module.EXPORT) == result_module.PASS


def test_analyze_exported_program_rejects_a_non_program():
    with pytest.raises(ExportFailed):
        analyze_exported_program(SoftmaxNet(), args=())


def test_live_and_pt2_paths_produce_equivalent_specs(tmp_path):
    """The same model through both doors must reach the same internal object."""
    from delegate_doctor import pt2_input
    from delegate_doctor.export_model import ModelSpec

    torch.manual_seed(0)
    model = SoftmaxNet().eval()
    args = (torch.randn(1, 4, 8, 8),)

    live_program = api.export_for_analysis(model, args=args)

    model_path = str(tmp_path / "model.pt2")
    inputs_path = str(tmp_path / "inputs.pt")
    torch.export.save(live_program, model_path)
    torch.save(args, inputs_path)
    artifact_spec = pt2_input.load_model_spec(model_path, inputs_path)

    live_spec = ModelSpec(name="live", exported_program=live_program,
                          example_args=args)

    assert isinstance(artifact_spec, ModelSpec) and isinstance(live_spec, ModelSpec)
    with torch.no_grad():
        assert torch.allclose(live_spec.call_baseline(), artifact_spec.call_baseline())


def test_repair_detection_is_identical_across_both_paths(tmp_path):
    from delegate_doctor import pt2_input
    from delegate_doctor.repairs import ALL_RULES

    torch.manual_seed(0)
    args = (torch.randn(1, 4, 8, 8),)
    live_program = api.export_for_analysis(SoftmaxNet().eval(), args=args)

    model_path = str(tmp_path / "model.pt2")
    inputs_path = str(tmp_path / "inputs.pt")
    torch.export.save(live_program, model_path)
    torch.save(args, inputs_path)
    artifact_spec = pt2_input.load_model_spec(model_path, inputs_path)

    for rule in ALL_RULES:
        live = rule.detect(copy.deepcopy(live_program))
        artifact = rule.detect(copy.deepcopy(artifact_spec.exported_program))
        assert live.applies == artifact.applies, rule.RULE_ID
        assert len(live.detections) == len(artifact.detections), rule.RULE_ID
