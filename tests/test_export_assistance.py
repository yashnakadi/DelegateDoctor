"""When export fails, help the interface the user wrote. Do not replace it.

The bug this file exists for, seen on Inception-V3:

    DelegateDoctor model interface  found
    Exporting...
    PyTorch export                  FAILED

    Preparing model with AI...
    No eligible class or existing model instance is exposed:
    delegate_doctor_model is a factory function...

DelegateDoctor had already found the interface, called it, and built a model.
Falling back to *generic* preparation then asked "which class is the model?"
about a file that had already answered, and rejected `delegate_doctor_model`
for being a function - which is exactly what the contract says it is.

So: a valid interface is authoritative. If export fails, the question becomes
"what small change makes this exportable?", the failure is preserved so
`--verbose` can show it, and the two interface functions are never replaced.

Fully offline. Real child processes where the test is about what actually
exports; fakes where it is about ordering and what crosses the boundary.
"""

import json
from pathlib import Path

import pytest
import torch

from delegate_doctor import cli, model_interface
from delegate_doctor.agent import export_assistance
from delegate_doctor.agent.export_assistance import (AssistanceValidationError,
                                                     ExportAdjustment)
from delegate_doctor.model_source import ModelSourceError

# --- sources --------------------------------------------------------------------

GOOD_SOURCE = '''
import torch


class Tiny(torch.nn.Module):
    def forward(self, x):
        return x + 1


def delegate_doctor_model():
    return Tiny()


def delegate_doctor_inputs():
    return (torch.randn(1, 4),)
'''

# An Inception-shaped model: a *factory*, construction succeeds, and export
# fails because the forward returns an auxiliary head alongside the logits.
INCEPTION_SHAPED = '''
import torch


class WithAuxHead(torch.nn.Module):
    """Stands in for Inception-V3: a real model whose forward returns extras."""

    def __init__(self, aux_logits=True):
        super().__init__()
        self.aux_logits = aux_logits
        self.stem = torch.nn.Linear(8, 8)
        self.aux = torch.nn.Linear(8, 8)

    def forward(self, x):
        primary = self.stem(x)
        if self.aux_logits:
            # Unconditional, so the fixture fails the same way every run. An
            # earlier version branched on the tensor's sum, which depends on
            # the child process's unseeded weight initialisation and reported
            # a different stage from one run to the next.
            raise RuntimeError("aux head is not exportable")
        return primary


def delegate_doctor_model():
    torch.manual_seed(0)
    model = WithAuxHead(aux_logits=True)
    model.eval()
    return model


def delegate_doctor_inputs():
    torch.manual_seed(0)
    return (torch.randn(1, 8),)
'''


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- the structured export failure ------------------------------------------------

def test_a_successful_interface_export_is_unchanged(tmp_path):
    """Case 1: nothing about this path moved."""
    model = write(tmp_path / "model.py", GOOD_SOURCE)
    prepared = model_interface.prepare_from_interface(
        model, tmp_path / "work", announce=lambda text: None)
    assert prepared.exported_program_path.is_file()


def test_a_successful_export_never_constructs_a_provider(tmp_path, monkeypatch):
    built = []
    monkeypatch.setattr("delegate_doctor.agent.client.build_provider",
                        lambda **kwargs: built.append(kwargs))

    model = write(tmp_path / "model.py", GOOD_SOURCE)
    cli.prepare_model_source(model, interactive=True,
                             announce=lambda text: None)
    assert built == []


def test_an_export_failure_is_retained_as_structured_data(tmp_path):
    """Case 2: the real exception survives, not just a category."""
    model = write(tmp_path / "model.py", INCEPTION_SHAPED)
    with pytest.raises(model_interface.ModelInterfaceError) as caught:
        model_interface.prepare_from_interface(
            model, tmp_path / "work", announce=lambda text: None)

    failure = caught.value.failure
    assert failure is not None
    assert failure.stage in (model_interface.STAGE_FORWARD,
                             model_interface.STAGE_EXPORT)
    assert failure.exception_type
    assert failure.message
    assert failure.traceback_text
    assert failure.is_export_stage


def test_the_failure_distinguishes_construction_from_export(tmp_path):
    """A model that cannot be built is a different answer from one that cannot export."""
    source = GOOD_SOURCE.replace("return Tiny()",
                                 "raise RuntimeError('no weights available')")
    model = write(tmp_path / "model.py", source)
    with pytest.raises(model_interface.ModelInterfaceError) as caught:
        model_interface.prepare_from_interface(
            model, tmp_path / "work", announce=lambda text: None)

    failure = caught.value.failure
    assert failure.stage == model_interface.STAGE_CONSTRUCTION
    assert not failure.is_export_stage, (
        "a construction failure was offered to export assistance")


def test_a_missing_dependency_is_not_an_export_failure(tmp_path):
    source = "import definitely_not_installed\n" + GOOD_SOURCE
    model = write(tmp_path / "model.py", source)
    with pytest.raises(model_interface.ModelInterfaceError) as caught:
        model_interface.prepare_from_interface(
            model, tmp_path / "work", announce=lambda text: None)
    assert caught.value.failure.stage == model_interface.STAGE_IMPORT
    assert not caught.value.failure.is_export_stage


# --- verbose shows the real reason ------------------------------------------------

def failing_export(monkeypatch, stage=None, message="aux head is not exportable"):
    failure = model_interface.ExportFailure(
        stage=stage or model_interface.STAGE_EXPORT,
        exception_type="RuntimeError", message=message,
        traceback_text='  File "model.py", line 21, in forward\n'
                       '    raise RuntimeError(...)')
    monkeypatch.setattr(
        model_interface, "prepare_from_interface",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            model_interface.ModelInterfaceError("export failed",
                                                failure=failure)))
    return failure


def test_verbose_shows_the_actual_export_exception(tmp_path, monkeypatch):
    """Case 3: not a category - the type, the message and the traceback."""
    failing_export(monkeypatch)
    monkeypatch.setattr("delegate_doctor.agent.client.build_provider",
                        lambda **kwargs: (_ for _ in ()).throw(
                            _no_provider()))

    said = []
    model = write(tmp_path / "model.py", INCEPTION_SHAPED)
    with pytest.raises(ModelSourceError):
        cli.prepare_model_source(model, interactive=True, allow_ai_source=True,
                                 announce=said.append, verbose=True)

    printed = "\n".join(said)
    assert "PyTorch export                  FAILED" in printed
    assert "Export failure" in printed
    assert "RuntimeError" in printed
    assert "aux head is not exportable" in printed
    assert "Traceback" in printed


def test_normal_mode_stays_concise(tmp_path, monkeypatch):
    """Case 13: the traceback is verbose-only."""
    failing_export(monkeypatch)
    monkeypatch.setattr("delegate_doctor.agent.client.build_provider",
                        lambda **kwargs: (_ for _ in ()).throw(_no_provider()))

    said = []
    model = write(tmp_path / "model.py", INCEPTION_SHAPED)
    with pytest.raises(ModelSourceError):
        cli.prepare_model_source(model, interactive=True, allow_ai_source=True,
                                 announce=said.append, verbose=False)

    printed = "\n".join(said)
    assert "PyTorch export                  FAILED" in printed
    assert "Traceback" not in printed
    assert "Export failure" not in printed


def test_the_traceback_is_sanitized(tmp_path):
    """Case 14: no home paths, no credentials."""
    import os

    home = os.path.expanduser("~")
    model = write(tmp_path / "model.py", INCEPTION_SHAPED)
    with pytest.raises(model_interface.ModelInterfaceError) as caught:
        model_interface.prepare_from_interface(
            model, tmp_path / "work", announce=lambda text: None)

    text = caught.value.failure.traceback_text
    assert home not in text
    assert "site-packages/" not in text or "/Users/" not in text


def _no_provider():
    from delegate_doctor.agent.client import AINotConfigured

    return AINotConfigured("AI NOT CONFIGURED")


# --- the interface stays authoritative ---------------------------------------------

class RecordingProvider:
    """Records what was sent and answers with a scripted adjustment."""

    configuration = None

    def __init__(self, *replies):
        self.requests = []
        self.replies = list(replies) or ["not an adjustment"]

    def complete_structured(self, request):
        from delegate_doctor.agent.client import AIResponse

        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.replies) - 1)
        return AIResponse(text=self.replies[index])

    @property
    def sent(self) -> str:
        return "\n".join(request.user for request in self.requests)


def adjustment_reply(**fields) -> str:
    return json.dumps(fields)


def test_generic_model_discovery_is_never_used_for_a_known_interface(
        tmp_path, monkeypatch):
    """Case 4: the whole point. The model is known; only the export is not."""
    failing_export(monkeypatch)
    provider = RecordingProvider(adjustment_reply(
        summary="drop the aux head", module_attributes={"aux_logits": False}))
    monkeypatch.setattr("delegate_doctor.agent.client.build_provider",
                        lambda **kwargs: provider)

    generic = []
    monkeypatch.setattr("delegate_doctor.agent.preparation.prepare_model",
                        lambda path, **kwargs: generic.append(kwargs))
    monkeypatch.setattr(
        export_assistance, "assist_export",
        lambda **kwargs: export_assistance.AssistanceOutcome(
            prepared="prepared",
            adjustment=ExportAdjustment(module_attributes={"aux_logits": False})))
    monkeypatch.setattr(model_interface, "model_spec_from_prepared",
                        lambda prepared, **kwargs: "spec")

    model = write(tmp_path / "model.py", INCEPTION_SHAPED)
    assert cli.prepare_model_source(model, interactive=False, allow_ai_source=True,
                                    announce=lambda text: None) == "spec"
    assert generic == [], "generic symbol discovery ran"


def test_a_model_factory_function_is_never_rejected_for_being_a_function(
        tmp_path, monkeypatch):
    """Case 5: `delegate_doctor_model` is a factory. That is the contract."""
    failing_export(monkeypatch)
    provider = RecordingProvider(adjustment_reply(
        module_attributes={"aux_logits": False}))
    monkeypatch.setattr("delegate_doctor.agent.client.build_provider",
                        lambda **kwargs: provider)
    monkeypatch.setattr(
        export_assistance, "assist_export",
        lambda **kwargs: export_assistance.AssistanceOutcome(
            prepared="prepared", adjustment=ExportAdjustment(
                module_attributes={"aux_logits": False})))
    monkeypatch.setattr(model_interface, "model_spec_from_prepared",
                        lambda prepared, **kwargs: "spec")

    said = []
    model = write(tmp_path / "model.py", INCEPTION_SHAPED)
    cli.prepare_model_source(model, interactive=False, allow_ai_source=True,
                             announce=said.append)

    printed = "\n".join(said)
    for banned in ("No eligible class", "existing model instance",
                   "is a factory function", "symbol_kind"):
        assert banned not in printed, banned


def test_the_assistance_prompt_names_the_existing_interface(tmp_path):
    """Case 6: the request says what the model is, and what failed."""
    failure = model_interface.ExportFailure(
        stage="export", exception_type="RuntimeError",
        message="aux head is not exportable",
        traceback_text="  File \"model.py\", line 21")
    prompt = export_assistance.build_prompt(
        "class WithAuxHead: ...", failure,
        model_interface.MODEL_FUNCTION, model_interface.INPUTS_FUNCTION,
        (model_interface.KWARGS_FUNCTION,))

    assert "delegate_doctor_model()" in prompt
    assert "delegate_doctor_inputs()" in prompt
    assert "RuntimeError" in prompt
    assert "aux head is not exportable" in prompt
    assert "class WithAuxHead" in prompt
    # And it forbids the question that caused the bug.
    assert "Do NOT propose a different model symbol" in prompt


def test_the_assistance_prompt_carries_nothing_private(tmp_path, monkeypatch):
    """Case 7: no weights, tensor values, environment or credentials."""
    monkeypatch.setenv("DELEGATE_DOCTOR_LLM_API_KEY", "sk-secret-value")
    failure = model_interface.ExportFailure(
        stage="export", exception_type="RuntimeError", message="nope")
    prompt = export_assistance.build_prompt(
        "class WithAuxHead: ...", failure, "delegate_doctor_model",
        "delegate_doctor_inputs")

    for forbidden in ("sk-secret-value", "state_dict", "Parameter containing",
                      "DELEGATE_DOCTOR_LLM_API_KEY", "PATH="):
        assert forbidden not in prompt, forbidden


# --- the adjustment schema ----------------------------------------------------------

def test_a_valid_adjustment_parses():
    adjustment = export_assistance.parse_adjustment({
        "summary": "drop the aux head",
        "module_attributes": {"aux_logits": False},
        "export_options": {"strict": False},
        "output_index": 0,
    })
    assert adjustment.module_attributes == {"aux_logits": False}
    assert adjustment.export_options == {"strict": False}
    assert adjustment.output_index == 0
    assert "aux_logits" in adjustment.describe()


@pytest.mark.parametrize("payload", [
    {"module_attributes": {"aux_logits": "False"}},        # a string
    {"module_attributes": {"__class__": 1}},               # private attribute
    {"module_attributes": {"not an identifier": 1}},
    {"export_options": {"exec": True}},                    # not allowlisted
    {"export_options": {"strict": "yes"}},                 # wrong type
    {"output_index": "first"},
    {"output_index": 99},
    {"unknown_field": 1},
    {},                                                    # proposes nothing
    "not an object",
])
def test_an_invalid_adjustment_is_refused(payload):
    """Case 9: rejected before anything is applied or executed."""
    with pytest.raises(AssistanceValidationError):
        export_assistance.parse_adjustment(payload)


@pytest.mark.parametrize("text", [
    "import os; os.system('rm -rf /')",
    "```python\nmodel.aux_logits = False\n```",
    "I think you should set aux_logits to False.",
])
def test_prose_and_code_are_refused(text):
    with pytest.raises(AssistanceValidationError):
        export_assistance.parse_adjustment_text(text)


def test_fenced_json_is_tolerated():
    adjustment = export_assistance.parse_adjustment_text(
        '```json\n{"module_attributes": {"aux_logits": false}}\n```')
    assert adjustment.module_attributes == {"aux_logits": False}


def test_the_schema_has_no_symbol_kind_constraint():
    """Case D: the class/existing_instance rule belongs to the generic path."""
    import inspect

    source = inspect.getsource(export_assistance)
    for token in ("symbol_kind", "existing_instance", "SYMBOL_KIND"):
        assert token not in source, token


def test_the_generic_preparation_schema_is_unchanged():
    """Case K/12: files with no interface keep the existing flow."""
    from delegate_doctor.agent import preparation_schema

    assert preparation_schema.ALLOWED_SYMBOL_KINDS == ("class",
                                                       "existing_instance")


# --- applying an adjustment, for real ------------------------------------------------

def test_an_adjustment_makes_the_real_interface_export(tmp_path):
    """Case 8: the change is applied and `torch.export` actually succeeds.

    A real child process, a real export, and the same interface functions -
    only the attribute the adjustment named is different.
    """
    model = write(tmp_path / "model.py", INCEPTION_SHAPED)

    with pytest.raises(model_interface.ModelInterfaceError):
        model_interface.prepare_from_interface(
            model, tmp_path / "work", announce=lambda text: None)

    prepared = model_interface.prepare_from_interface(
        model, tmp_path / "work2", announce=lambda text: None,
        adjustment=ExportAdjustment(module_attributes={"aux_logits": False}))
    assert prepared.exported_program_path.is_file()

    spec = model_interface.model_spec_from_prepared(prepared)
    assert spec.exported_program is not None


def test_an_adjustment_naming_an_absent_attribute_fails_clearly(tmp_path):
    model = write(tmp_path / "model.py", GOOD_SOURCE)
    with pytest.raises(model_interface.ModelInterfaceError) as caught:
        model_interface.prepare_from_interface(
            model, tmp_path / "work", announce=lambda text: None,
            adjustment=ExportAdjustment(module_attributes={"nonexistent": 1}))
    assert "attribute" in str(caught.value).lower()


def test_the_output_index_wrapper_keeps_the_interface(tmp_path):
    """A multi-output forward is narrowed by DelegateDoctor's own wrapper."""
    source = GOOD_SOURCE.replace("return x + 1", "return x + 1, x * 2")
    model = write(tmp_path / "model.py", source)

    prepared = model_interface.prepare_from_interface(
        model, tmp_path / "work", announce=lambda text: None,
        adjustment=ExportAdjustment(output_index=0))
    spec = model_interface.model_spec_from_prepared(prepared)
    assert spec.exported_program is not None


# --- the assistance loop ---------------------------------------------------------------

def fake_export(results):
    """An `export` double that returns or raises per scripted result."""
    calls = []

    def export(model_path, workspace, adjustment=None, quiet=False):
        calls.append(adjustment)
        outcome = results[min(len(calls) - 1, len(results) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    export.calls = calls
    return export


def failure_for(message="nope"):
    return model_interface.ExportFailure(
        stage="export", exception_type="RuntimeError", message=message)


def test_a_valid_adjustment_is_applied_and_retried():
    provider = RecordingProvider(adjustment_reply(
        module_attributes={"aux_logits": False}))
    export = fake_export(["prepared"])

    outcome = export_assistance.assist_export(
        model_path=Path("model.py"), workspace=Path("/tmp"),
        failure=failure_for(), provider=provider, source_text="src",
        export=export, announce=lambda text: None)

    assert outcome.succeeded
    assert outcome.adjustment.module_attributes == {"aux_logits": False}
    assert len(provider.requests) == 1
    assert export.calls[0].module_attributes == {"aux_logits": False}


def test_an_invalid_suggestion_never_reaches_the_export():
    """Case 9: rejected before execution."""
    provider = RecordingProvider("not an adjustment at all",
                                 "still not an adjustment")
    export = fake_export(["prepared"])

    outcome = export_assistance.assist_export(
        model_path=Path("model.py"), workspace=Path("/tmp"),
        failure=failure_for(), provider=provider, source_text="src",
        export=export, announce=lambda text: None)

    assert not outcome.succeeded
    assert export.calls == [], "an invalid adjustment was applied"
    assert outcome.reason == export_assistance.ASSISTANCE_NO_CHANGE


def test_repeated_failure_is_bounded():
    """Case 10: two attempts, then a clear final failure."""
    provider = RecordingProvider(
        adjustment_reply(module_attributes={"aux_logits": False}),
        adjustment_reply(export_options={"strict": False}))
    export = fake_export([
        model_interface.ModelInterfaceError("still failing",
                                            failure=failure_for("again")),
        model_interface.ModelInterfaceError("still failing",
                                            failure=failure_for("again")),
    ])

    outcome = export_assistance.assist_export(
        model_path=Path("model.py"), workspace=Path("/tmp"),
        failure=failure_for(), provider=provider, source_text="src",
        export=export, announce=lambda text: None)

    assert not outcome.succeeded
    assert len(provider.requests) == \
        export_assistance.MAX_EXPORT_ASSISTANCE_ATTEMPTS
    assert outcome.reason == export_assistance.ASSISTANCE_STILL_FAILING


def test_the_second_attempt_is_told_what_failed():
    provider = RecordingProvider(
        "garbage",
        adjustment_reply(module_attributes={"aux_logits": False}))
    export = fake_export(["prepared"])

    outcome = export_assistance.assist_export(
        model_path=Path("model.py"), workspace=Path("/tmp"),
        failure=failure_for(), provider=provider, source_text="src",
        export=export, announce=lambda text: None)

    assert outcome.succeeded
    assert "not usable" in provider.requests[1].user


def test_a_provider_error_stops_cleanly():
    from delegate_doctor.agent.client import AIError

    class Broken:
        configuration = None

        def complete_structured(self, request):
            raise AIError("provider is down")

    outcome = export_assistance.assist_export(
        model_path=Path("model.py"), workspace=Path("/tmp"),
        failure=failure_for(), provider=Broken(), source_text="src",
        export=fake_export(["prepared"]), announce=lambda text: None)

    assert not outcome.succeeded
    assert export_assistance.ASSISTANCE_UNAVAILABLE in outcome.reason


# --- pretrained weights ------------------------------------------------------------------

def test_a_successful_construction_never_asks_for_a_checkpoint_filename(
        tmp_path, monkeypatch):
    """Case 11: the model exists. A checkpoint path is not the problem.

    The old generic path demanded a local checkpoint whenever a factory used a
    torchvision weights enum. Construction had already succeeded, so that
    demand was about a step that had visibly worked.
    """
    failing_export(monkeypatch)
    provider = RecordingProvider(adjustment_reply(
        module_attributes={"aux_logits": False}))
    monkeypatch.setattr("delegate_doctor.agent.client.build_provider",
                        lambda **kwargs: provider)
    monkeypatch.setattr(
        export_assistance, "assist_export",
        lambda **kwargs: export_assistance.AssistanceOutcome(
            prepared="prepared", adjustment=ExportAdjustment(
                module_attributes={"aux_logits": False})))
    monkeypatch.setattr(model_interface, "model_spec_from_prepared",
                        lambda prepared, **kwargs: "spec")

    weights_source = INCEPTION_SHAPED.replace(
        "WithAuxHead(aux_logits=True)",
        "WithAuxHead(aux_logits=True)  # weights=Inception_V3_Weights.DEFAULT")
    said = []
    model = write(tmp_path / "model.py", weights_source)
    cli.prepare_model_source(model, interactive=False, allow_ai_source=True,
                             announce=said.append)

    printed = "\n".join(said)
    # The privacy disclosure legitimately lists "checkpoint contents" among
    # the things never sent. What must not appear is a *demand* for one.
    for banned in ("local checkpoint filename", "cached download",
                   "no local checkpoint", "checkpoint not found"):
        assert banned not in printed.lower(), banned


# --- what the user is told at the end ------------------------------------------------------

def test_an_applied_change_is_reported_concisely(tmp_path, monkeypatch):
    """Case G: what changed, without chain-of-thought."""
    failing_export(monkeypatch)
    monkeypatch.setattr("delegate_doctor.agent.client.build_provider",
                        lambda **kwargs: RecordingProvider())
    monkeypatch.setattr(
        export_assistance, "assist_export",
        lambda **kwargs: export_assistance.AssistanceOutcome(
            prepared="prepared",
            adjustment=ExportAdjustment(summary="drop the aux head",
                                        module_attributes={"aux_logits": False})))
    monkeypatch.setattr(model_interface, "model_spec_from_prepared",
                        lambda prepared, **kwargs: "spec")

    said = []
    model = write(tmp_path / "model.py", INCEPTION_SHAPED)
    cli.prepare_model_source(model, interactive=False, allow_ai_source=True,
                             announce=said.append)

    printed = "\n".join(said)
    assert "Export assistance       APPLIED" in printed
    assert "aux_logits -> False" in printed
    assert "PyTorch export                  PASS" in printed


@pytest.mark.parametrize("reason, expected", [
    (export_assistance.ASSISTANCE_NO_CHANGE, "no valid change"),
    (export_assistance.ASSISTANCE_STILL_FAILING, "still did not export"),
])
def test_the_failure_categories_are_distinguished(reason, expected, tmp_path,
                                                  monkeypatch):
    """Case J: not all collapsed into AI PREPARATION NEEDS INPUT."""
    failing_export(monkeypatch)
    monkeypatch.setattr("delegate_doctor.agent.client.build_provider",
                        lambda **kwargs: RecordingProvider())
    monkeypatch.setattr(
        export_assistance, "assist_export",
        lambda **kwargs: export_assistance.AssistanceOutcome(reason=reason))

    model = write(tmp_path / "model.py", INCEPTION_SHAPED)
    with pytest.raises(ModelSourceError) as caught:
        cli.prepare_model_source(model, interactive=False, allow_ai_source=True,
                                 announce=lambda text: None)

    message = str(caught.value)
    assert expected in message
    assert "AI PREPARATION NEEDS INPUT" not in message


def test_assistance_unavailable_is_its_own_answer(tmp_path, monkeypatch):
    """Case J.3: no provider is a different thing from a failed adjustment."""
    failing_export(monkeypatch)
    monkeypatch.setattr("delegate_doctor.agent.client.build_provider",
                        lambda **kwargs: (_ for _ in ()).throw(_no_provider()))

    model = write(tmp_path / "model.py", INCEPTION_SHAPED)
    with pytest.raises(ModelSourceError) as caught:
        cli.prepare_model_source(model, interactive=False, allow_ai_source=True,
                                 announce=lambda text: None)
    assert "AI preparation is unavailable" in str(caught.value)


# --- the Inception regression, end to end -------------------------------------------------

def test_the_inception_shape_is_assisted_rather_than_rediscovered(tmp_path,
                                                                  monkeypatch):
    """The reported case, with a lightweight stand-in for Inception-V3.

    Interface exists, construction succeeds, export fails. DelegateDoctor must
    keep the interface, preserve the error, assist *that* interface, and retry
    the same two functions after a validated adjustment.
    """
    provider = RecordingProvider(adjustment_reply(
        summary="disable the auxiliary head",
        module_attributes={"aux_logits": False}))
    monkeypatch.setattr("delegate_doctor.agent.client.build_provider",
                        lambda **kwargs: provider)

    generic = []
    monkeypatch.setattr("delegate_doctor.agent.preparation.prepare_model",
                        lambda path, **kwargs: generic.append(kwargs))

    said = []
    model = write(tmp_path / "model.py", INCEPTION_SHAPED)
    spec = cli.prepare_model_source(model, interactive=False, allow_ai_source=True,
                                    announce=said.append, verbose=True)

    printed = "\n".join(said)

    # The interface was found and kept.
    assert "DelegateDoctor model interface  found" in printed
    assert generic == [], "generic class discovery ran"
    assert "is a factory function" not in printed

    # The real export failure was preserved and shown.
    assert "Export failure" in printed
    assert "RuntimeError" in printed

    # Assistance was asked about *this* interface.
    assert "delegate_doctor_model()" in provider.sent
    assert "delegate_doctor_inputs()" in provider.sent
    assert "Do NOT propose a different model symbol" in provider.sent

    # And the same interface exported after the adjustment.
    assert "aux_logits -> False" in printed
    assert spec.exported_program is not None
    assert len(spec.example_args) == 1


# --- AI preparation and export assistance are independent of --ai-repair ---------

def test_preparation_works_without_the_ai_repair_flag(tmp_path, monkeypatch):
    """Case 7: getting the model into torch.export is not an experiment.

    `--ai-repair` gates optimization-time repair. Preparation solves a
    model-loading problem, and gating it behind the same switch would make a
    model unanalyzable unless the user also opted into an unrelated feature.
    """
    reached = []
    monkeypatch.setattr("delegate_doctor.agent.client.build_provider",
                        lambda **kwargs: object())
    monkeypatch.setattr("delegate_doctor.agent.preparation.prepare_model",
                        lambda path, **kwargs: reached.append(kwargs) or object())
    monkeypatch.setattr("delegate_doctor.agent.preparation.model_spec_from_outcome",
                        lambda outcome, **kwargs: "spec")

    model = write(tmp_path / "model.py", "import torch\n")
    # No ai_repair anywhere in this call.
    assert cli.prepare_model_source(model, interactive=True,
                                    announce=lambda text: None) == "spec"
    assert reached, "AI preparation was blocked without --ai-repair"


def test_export_assistance_works_without_the_ai_repair_flag(tmp_path,
                                                            monkeypatch):
    """Case 8."""
    failing_export(monkeypatch)
    monkeypatch.setattr("delegate_doctor.agent.client.build_provider",
                        lambda **kwargs: RecordingProvider())
    monkeypatch.setattr(
        export_assistance, "assist_export",
        lambda **kwargs: export_assistance.AssistanceOutcome(
            prepared="prepared",
            adjustment=ExportAdjustment(module_attributes={"aux_logits": False})))
    monkeypatch.setattr(model_interface, "model_spec_from_prepared",
                        lambda prepared, **kwargs: "spec")

    model = write(tmp_path / "model.py", INCEPTION_SHAPED)
    # Source consent is still asked for - assistance sends selected source -
    # and that is a different question from enabling AI repair.
    assert cli.prepare_model_source(model, interactive=True,
                                    announce=lambda text: None,
                                    prompt=lambda question: "y") == "spec"


def test_the_ai_repair_flag_does_not_authorize_a_source_upload(tmp_path,
                                                               monkeypatch):
    """Case 9: source consent is a separate privacy boundary.

    `prepare_model_source` takes no `ai_repair` argument at all, which is the
    strongest possible form of "the flag cannot reach here".
    """
    import inspect

    signature = inspect.signature(cli.prepare_model_source)
    assert "ai_repair" not in signature.parameters
    assert "allow_ai_source" in signature.parameters


def test_source_consent_is_still_required_non_interactively(tmp_path,
                                                            monkeypatch):
    """Case 9/10: --ai-repair grants nothing here."""
    from delegate_doctor.model_source import ModelSourceError

    built = []
    monkeypatch.setattr("delegate_doctor.agent.client.build_provider",
                        lambda **kwargs: built.append(kwargs))

    model = write(tmp_path / "model.py", "import torch\n")
    with pytest.raises(ModelSourceError) as caught:
        cli.prepare_model_source(model, interactive=False,
                                 allow_ai_source=False,
                                 announce=lambda text: None)
    assert "--allow-ai" in str(caught.value)
    assert built == [], "a provider was built without source consent"


def test_the_two_flags_control_different_capabilities():
    """Case 16: no single switch does both any more."""
    import inspect

    optimize = inspect.signature(cli.run_optimize).parameters
    assert "ai_repair" in optimize
    assert "allow_ai_source" in optimize
    assert optimize["ai_repair"].default is False
    assert optimize["allow_ai_source"].default is False
    # And the old combined name is gone.
    assert "allow_ai" not in optimize
