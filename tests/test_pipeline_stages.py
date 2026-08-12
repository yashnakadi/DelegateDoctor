"""Stage outcomes: how far a model got, and why it stopped there.

The point of these tests is the distinction the product rests on - "this stage
could not run" is not "this model failed". Device work is mocked throughout;
nothing here needs adb, a target or a runner binary.
"""

import pytest
import torch

from delegate_doctor import capabilities, pipeline, result as result_module
from delegate_doctor.export_model import ModelSpec
from delegate_doctor.result import OptimizationResult, Stage


# --- fixtures ---------------------------------------------------------------

class SoftmaxNet(torch.nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=1)


class PlainNet(torch.nn.Module):
    """Fully delegable: a single addition XNNPACK is happy to take."""

    def forward(self, x):
        return x + x


class UnrepairableNet(torch.nn.Module):
    """Lowers fine, has a portable fallback, but matches no catalog rule."""

    def forward(self, x):
        return torch.fmod(x, 2.0)


def spec_for(model, args, name="test model"):
    return ModelSpec(
        name=name,
        exported_program=torch.export.export(model.eval(), args),
        example_args=args,
    )


def run(spec, tmp_path, **options):
    options.setdefault("artifacts_dir", str(tmp_path / "art"))
    options.setdefault("quiet", True)
    return pipeline.run_optimization(spec, **options)


@pytest.fixture
def no_device(monkeypatch):
    monkeypatch.setattr(pipeline, "_find_device",
                        lambda runners_dir: (None, None, None,
                                             "No Arm64 Android target is attached."))


# --- ExecuTorch lowering failure (acceptance criterion C) -------------------

def test_a_lowering_failure_is_reported_as_an_executorch_limit(tmp_path, monkeypatch,
                                                               no_device):
    """Export passed. ExecuTorch is what declined, and the result must say so."""
    def explode(*args, **kwargs):
        raise RuntimeError("edge dialect rejected aten.exotic_op")

    monkeypatch.setattr(pipeline.export_model, "lower_and_write", explode)

    outcome = run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)

    assert outcome.status == result_module.EXECUTORCH_LOWERING_UNSUPPORTED
    assert outcome.stage_status(result_module.EXPORT) == result_module.PASS
    assert outcome.stage_status(result_module.LOWERING) == result_module.FAILED
    assert "exotic_op" in outcome.stage(result_module.LOWERING).detail
    # The report must not blame torch.export.
    assert "PYTORCH EXPORT FAILED" not in outcome.report_text
    assert "ExecuTorch could not lower" in outcome.report_text


def test_a_lowering_failure_does_not_raise(tmp_path, monkeypatch, no_device):
    monkeypatch.setattr(pipeline.export_model, "lower_and_write",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope")))
    outcome = run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)
    assert isinstance(outcome, OptimizationResult)
    assert outcome.exit_code == 0        # a finding, not a tool error


# --- no device (acceptance criterion D) -------------------------------------

def test_static_analysis_survives_a_missing_device(tmp_path, no_device):
    outcome = run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)

    assert outcome.stage_status(result_module.LOWERING) == result_module.PASS
    assert outcome.stage_status(result_module.DELEGATION) == result_module.PASS
    assert outcome.stage_status(result_module.DEVICE) == result_module.UNAVAILABLE
    assert outcome.before_delegation is not None
    assert outcome.status == result_module.ANALYSIS_COMPLETE


def test_a_matched_rule_is_not_applied_without_a_device(tmp_path, no_device):
    """Detection is static; acceptance needs the benchmark. Keep them apart."""
    outcome = run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)

    assert outcome.repair_available
    assert outcome.repairs_applied == {}
    assert outcome.output_pte is None
    assert "not applied" in outcome.stage(result_module.REPAIR).detail.lower()


def test_no_stage_claims_pass_without_running(tmp_path, no_device):
    outcome = run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)
    for name in (result_module.PROFILING, result_module.VERIFICATION,
                 result_module.BENCHMARK):
        assert outcome.stage_status(name) == result_module.NOT_RUN
    assert outcome.host_verification is None
    assert outcome.device_verification is None
    assert outcome.benchmark is None
    assert outcome.decision is None


def test_the_report_never_prints_pass_for_a_stage_that_did_not_run(tmp_path, no_device):
    outcome = run(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)
    for line in outcome.report_text.splitlines():
        if line.startswith(("Runtime profiling", "Device benchmark",
                            "Correctness verification")):
            assert "PASS" not in line, line


# --- fully delegated --------------------------------------------------------

def test_a_fully_delegated_model_needs_no_repair(tmp_path, no_device):
    outcome = run(spec_for(PlainNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)
    assert outcome.before_delegation.portable_op_total == 0
    assert outcome.status == result_module.FULLY_DELEGATED
    assert outcome.exit_code == 0
    assert not outcome.repair_available


# --- no repair available (acceptance criterion B) ---------------------------

class FakeDevice:
    serial = "test-target"
    is_emulator = False

    def short_description(self):
        return "TestTarget · arm64-v8a · Android 35"

    def describe(self):
        return "Arm64 Android device - TestTarget"


def fake_profile(portable_ms=8.2, kernels=("native_call_fmod.out",)):
    """A ProfileResult shaped like a real one, without touching a device."""
    from delegate_doctor.profiling import PortableKernel, ProfileResult

    return ProfileResult(
        method_execute_ms=20.0,
        delegated_ms=20.0 - portable_ms,
        portable_ms=portable_ms,
        delegate_call_count=1,
        operator_call_count=len(kernels),
        portable_kernels=[
            PortableKernel(name=name, total_ms=portable_ms,
                           call_count=1, runtime_fraction=portable_ms / 20.0)
            for name in kernels
        ],
    )


@pytest.fixture
def fake_target(monkeypatch):
    """A device that exists and profiles, so the no-repair branch is reachable."""
    monkeypatch.setattr(pipeline, "_find_device",
                        lambda runners_dir: (FakeDevice(), "bench", "etdump", ""))
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        lambda **kwargs: fake_profile())


def test_a_model_with_no_matching_rule_is_still_analyzed(tmp_path, fake_target):
    """Export, lowering and profiling all worked. There is simply no rule."""
    outcome = run(spec_for(UnrepairableNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)

    assert outcome.status == result_module.NO_REPAIR_AVAILABLE
    assert outcome.exit_code == 0
    assert outcome.stage_status(result_module.PROFILING) == result_module.PASS
    assert outcome.stage_status(result_module.REPAIR) == result_module.NONE_FOUND
    assert not outcome.repair_available
    assert outcome.before_profile is not None
    # The user still learns where the time went.
    assert "HOTSPOTS" in outcome.report_text
    assert outcome.output_pte is None


def test_no_portable_hotspot_means_no_repair_required(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "_find_device",
                        lambda runners_dir: (FakeDevice(), "bench", "etdump", ""))
    monkeypatch.setattr(pipeline.profiling, "profile_model",
                        lambda **kwargs: fake_profile(portable_ms=0.0, kernels=()))

    outcome = run(spec_for(PlainNet(), (torch.randn(1, 4, 8, 8),)), tmp_path)
    assert outcome.status == result_module.NO_REPAIR_REQUIRED
    assert outcome.exit_code == 0


def test_an_unverifiable_output_blocks_acceptance(tmp_path, fake_target):
    """A repair that cannot be checked is never accepted."""
    class TwoOutputSoftmax(torch.nn.Module):
        def forward(self, x):
            return torch.softmax(x, dim=1), x * 2

    outcome = run(spec_for(TwoOutputSoftmax(), (torch.randn(1, 4, 8, 8),)), tmp_path)

    assert outcome.repair_available          # DD-001 did match
    assert outcome.stage_status(result_module.VERIFICATION) == result_module.UNSUPPORTED
    assert outcome.status == result_module.DEVICE_EXECUTION_UNSUPPORTED
    assert outcome.output_pte is None        # nothing was published
    assert outcome.decision is None
    assert "will not accept a repair it cannot verify" in outcome.report_text


# --- device transport limits ------------------------------------------------

def test_an_int64_input_is_a_transport_limit_not_a_rejection(tmp_path, monkeypatch):
    """A device is attached, but the runner cannot carry this dtype."""
    monkeypatch.setattr(pipeline, "_find_device",
                        lambda runners_dir: (FakeDevice(), "bench", "etdump", ""))

    class IntNet(torch.nn.Module):
        def forward(self, ids):
            return ids.float() + 1.0

    outcome = run(spec_for(IntNet(), (torch.randint(0, 5, (1, 4)),)), tmp_path)

    assert outcome.stage_status(result_module.EXPORT) == result_module.PASS
    assert outcome.stage_status(result_module.LOWERING) == result_module.PASS
    assert outcome.stage_status(result_module.DEVICE) == result_module.UNSUPPORTED
    assert "int64" in outcome.stage(result_module.DEVICE).detail
    assert outcome.status == result_module.DEVICE_EXECUTION_UNSUPPORTED
    assert outcome.benchmark is None


def test_kwargs_are_a_transport_limit_not_a_rejection():
    capability = capabilities.assess_inputs((torch.randn(1, 4),),
                                            {"attention_mask": torch.randn(1, 4)})
    assert not capability
    assert "keyword arguments" in capability.reason


def test_supported_inputs_pass_the_transport_check():
    assert capabilities.assess_inputs((torch.randn(1, 4),), {})


@pytest.mark.parametrize("bad", [
    torch.randint(0, 5, (1, 4)),
    torch.zeros(1, 4, dtype=torch.bool),
    torch.zeros(1, 4, dtype=torch.float16),
])
def test_non_fp32_inputs_are_named_precisely(bad):
    capability = capabilities.assess_inputs((bad,), {})
    assert not capability
    assert str(bad.dtype) in capability.reason


def test_non_tensor_arguments_are_named():
    capability = capabilities.assess_inputs((torch.randn(1, 4), 4), {})
    assert not capability
    assert "input 1 is int" in capability.reason


# --- output structure -------------------------------------------------------

def test_a_single_fp32_tensor_output_is_verifiable():
    assert capabilities.assess_output(torch.randn(1, 4))


def test_a_tuple_of_one_tensor_is_verifiable():
    assert capabilities.assess_output((torch.randn(1, 4),))


def test_multiple_outputs_report_verification_as_unsupported():
    capability = capabilities.assess_output((torch.randn(1, 4), torch.randn(1, 4)))
    assert not capability
    assert "2 outputs" in capability.reason
    assert "first tensor only" in capability.reason


def test_a_dict_output_is_unsupported():
    capability = capabilities.assess_output({"logits": torch.randn(1, 4)})
    assert not capability
    assert "dict" in capability.reason


def test_a_non_fp32_output_is_unsupported():
    capability = capabilities.assess_output(torch.randint(0, 5, (1, 4)))
    assert not capability
    assert "int64" in capability.reason


def test_first_output_tensor_picks_the_leaf():
    tensor = torch.randn(1, 4)
    assert capabilities.first_output_tensor(tensor) is tensor
    assert capabilities.first_output_tensor((tensor, 1)) is tensor
    assert capabilities.first_output_tensor({"a": tensor}) is None


# --- the result object ------------------------------------------------------

def test_stages_are_kept_in_pipeline_order():
    outcome = OptimizationResult(status=result_module.ANALYSIS_COMPLETE)
    outcome.record(result_module.BENCHMARK, result_module.NOT_RUN)
    outcome.record(result_module.EXPORT, result_module.PASS)
    outcome.record(result_module.LOWERING, result_module.PASS)
    assert [s.name for s in outcome.stages] == [
        result_module.EXPORT, result_module.LOWERING, result_module.BENCHMARK,
    ]


def test_recording_a_stage_twice_replaces_it():
    outcome = OptimizationResult(status=result_module.ANALYSIS_COMPLETE)
    outcome.record(result_module.DEVICE, result_module.PASS)
    outcome.record(result_module.DEVICE, result_module.UNSUPPORTED, "int64")
    assert len([s for s in outcome.stages if s.name == result_module.DEVICE]) == 1
    assert outcome.stage(result_module.DEVICE).status == result_module.UNSUPPORTED


def test_only_a_rejected_repair_exits_nonzero():
    """Every other outcome is a successful analysis."""
    assert result_module.exit_code(result_module.REPAIR_REJECTED) == 1
    for status in (result_module.ANALYSIS_COMPLETE, result_module.FULLY_DELEGATED,
                   result_module.NO_REPAIR_AVAILABLE,
                   result_module.NO_REPAIR_REQUIRED,
                   result_module.REPAIR_ACCEPTED,
                   result_module.EXECUTORCH_LOWERING_UNSUPPORTED,
                   result_module.DEVICE_EXECUTION_UNSUPPORTED):
        assert result_module.exit_code(status) == 0, status


def test_every_outcome_has_a_human_summary():
    for status in (result_module.ANALYSIS_COMPLETE, result_module.FULLY_DELEGATED,
                   result_module.NO_REPAIR_AVAILABLE,
                   result_module.NO_REPAIR_REQUIRED,
                   result_module.REPAIR_ACCEPTED, result_module.REPAIR_REJECTED,
                   result_module.EXECUTORCH_LOWERING_UNSUPPORTED,
                   result_module.DEVICE_EXECUTION_UNSUPPORTED):
        assert result_module.OUTCOME_TEXT[status]


def test_a_stage_knows_whether_it_ran():
    assert Stage("x", result_module.PASS).ran
    assert Stage("x", result_module.FAILED).ran
    assert not Stage("x", result_module.NOT_RUN).ran
    assert not Stage("x", result_module.UNAVAILABLE).ran


# --- the console stays clean -------------------------------------------------

NOISE_MARKERS = (
    "LeafSpec",
    "Output Buffer not found",
    "Redirects are currently not supported",
    "KernelPreference",
    "register_constant()",
    "cpuinfo_utils.cpp",
)


def test_a_normal_run_prints_progress_and_nothing_upstream(tmp_path, capsys,
                                                           no_device):
    """A representative run: intentional progress, then the summary."""
    spec = spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),))
    pipeline.run_optimization(spec, artifacts_dir=str(tmp_path / "art"))

    captured = capsys.readouterr()
    output = captured.out + captured.err

    assert "Lowering with XNNPACK..." in output
    assert "DelegateDoctor - test model" in output
    assert "Result" in output
    for marker in NOISE_MARKERS:
        assert marker not in output, f"upstream noise reached the console: {marker}"


def test_the_known_warnings_are_filtered_during_a_run(tmp_path, no_device):
    """The suppression is actually active around the pipeline body."""
    import warnings

    from delegate_doctor import console_noise

    spec = spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.run_optimization(spec, artifacts_dir=str(tmp_path / "art"),
                                  quiet=True)

    leaked = [w for w in caught
              if console_noise.is_known_warning(w.category, str(w.message))]
    assert leaked == [], f"{len(leaked)} known-benign warnings leaked"


def test_verbose_keeps_the_upstream_warnings(tmp_path, no_device):
    """--verbose must not silently apply the normal suppression policy."""
    import warnings

    from delegate_doctor import console_noise

    spec = spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.run_optimization(spec, artifacts_dir=str(tmp_path / "art"),
                                  verbose=True, quiet=True)

    known = [w for w in caught
             if console_noise.is_known_warning(w.category, str(w.message))]
    assert known, "verbose mode hid the upstream warnings"


def test_quiet_mode_is_not_a_stderr_black_hole(tmp_path, capsys, no_device):
    """quiet silences DelegateDoctor's reporting, not real problems."""
    import warnings

    spec = spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.run_optimization(spec, artifacts_dir=str(tmp_path / "art"),
                                  quiet=True)
        warnings.warn("a real problem after the run", UserWarning)

    assert "Lowering with XNNPACK" not in capsys.readouterr().out
    assert any("a real problem" in str(w.message) for w in caught)


def test_the_run_restores_the_callers_warning_configuration(tmp_path, no_device):
    import warnings

    before = list(warnings.filters)
    pipeline.run_optimization(spec_for(SoftmaxNet(), (torch.randn(1, 4, 8, 8),)),
                              artifacts_dir=str(tmp_path / "art"), quiet=True)
    assert warnings.filters == before
