"""Helping an existing model interface export. Never rediscovering the model.

When a file declares

    def delegate_doctor_model():
        ...
    def delegate_doctor_inputs():
        ...

the user has already said how their model is built. If `torch.export` then
refuses it, the useful question is "what small change makes *this* exportable?"
- not "which class in this file is the model?", which DelegateDoctor already
knows and which the generic preparation path exists to answer for files that
do not declare an interface.

The distinction matters because the generic path would look at the same file
and report:

    No eligible class or existing model instance is exposed:
    delegate_doctor_model is a factory function...

which is both wrong and confusing: the factory was found, was called, and
produced a model. Only the export failed.

So this is a separate path with a separate schema. It accepts a *factory* as
the source of the model, because that is what the interface is, and it may
only propose adjustments to a model that already exists:

    module_attributes    set an attribute on the constructed model
    export_options       an allowlisted torch.export keyword
    output_index         keep one output of a multi-output forward

That is the whole vocabulary. It is small on purpose: every entry is a literal
that DelegateDoctor applies itself, in its own adapter, in the sanitized child
process. No Python from the provider is ever executed, and the user's public
contract - the two interface functions - is never replaced.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

# How many adjustments to try. Each costs one provider request and one export,
# and a third guess is rarely a better guess.
MAX_EXPORT_ASSISTANCE_ATTEMPTS = 2

# `torch.export` keywords DelegateDoctor will let an adjustment set. Anything
# that changes what the model *means* is absent by construction.
ALLOWED_EXPORT_OPTIONS = {
    "strict": bool,
}

MAX_ATTRIBUTES = 6
MAX_SUMMARY_LENGTH = 200
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class AssistanceValidationError(ValueError):
    """The proposed adjustment is not something DelegateDoctor will apply."""


@dataclass
class ExportAdjustment:
    """A validated, bounded change to an existing interface's export.

    Everything here is data. The adapter that applies it is DelegateDoctor's,
    and every value has been through `_check_literal` before it can reach a
    `repr()` in generated source.
    """

    summary: str = ""
    module_attributes: dict = field(default_factory=dict)
    export_options: dict = field(default_factory=dict)
    output_index: Optional[int] = None
    reasoning_absent: bool = True

    @property
    def is_empty(self) -> bool:
        return not (self.module_attributes or self.export_options
                    or self.output_index is not None)

    def describe(self) -> str:
        """A concise, sanitized account of what changed. Never chain-of-thought."""
        parts = []
        for name, value in sorted(self.module_attributes.items()):
            parts.append(f"{name} -> {value!r}")
        for name, value in sorted(self.export_options.items()):
            parts.append(f"export {name}={value!r}")
        if self.output_index is not None:
            parts.append(f"keep output[{self.output_index}]")
        return ", ".join(parts) or "no change"

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "module_attributes": dict(self.module_attributes),
            "export_options": dict(self.export_options),
            "output_index": self.output_index,
        }


def _check_literal(value, where: str):
    """Only small plain data. Never a string that could be code or a path."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if abs(value) > 2 ** 31:
            raise AssistanceValidationError(f"{where} integer out of range")
        return value
    if isinstance(value, float):
        if value != value or abs(value) == float("inf"):
            raise AssistanceValidationError(f"{where} must be finite")
        return value
    if isinstance(value, str):
        # A string is where a path, a URL or an import would hide. None of the
        # adjustments here needs one.
        raise AssistanceValidationError(
            f"{where} may not be a string. Adjustments are numbers, booleans "
            f"or null.")
    raise AssistanceValidationError(
        f"{where} has unsupported type {type(value).__name__}")


def parse_adjustment(payload) -> ExportAdjustment:
    """Validate a proposed adjustment, or refuse it. Nothing is executed."""
    if not isinstance(payload, dict):
        raise AssistanceValidationError("the adjustment must be a JSON object")

    unknown = set(payload) - {"summary", "module_attributes",
                              "export_options", "output_index"}
    if unknown:
        raise AssistanceValidationError(
            f"unknown field(s): {', '.join(sorted(unknown))}")

    attributes = payload.get("module_attributes") or {}
    if not isinstance(attributes, dict):
        raise AssistanceValidationError("module_attributes must be an object")
    if len(attributes) > MAX_ATTRIBUTES:
        raise AssistanceValidationError(
            f"at most {MAX_ATTRIBUTES} module_attributes may be set")
    checked_attributes = {}
    for name, value in attributes.items():
        if not isinstance(name, str) or not _IDENTIFIER.match(name):
            raise AssistanceValidationError(
                f"module_attributes key {name!r} is not a plain identifier")
        if name.startswith("_"):
            raise AssistanceValidationError(
                f"module_attributes may not set the private attribute {name!r}")
        checked_attributes[name] = _check_literal(
            value, f"module_attributes[{name!r}]")

    options = payload.get("export_options") or {}
    if not isinstance(options, dict):
        raise AssistanceValidationError("export_options must be an object")
    checked_options = {}
    for name, value in options.items():
        if name not in ALLOWED_EXPORT_OPTIONS:
            raise AssistanceValidationError(
                f"export option {name!r} is not allowlisted. Allowed: "
                f"{', '.join(sorted(ALLOWED_EXPORT_OPTIONS))}")
        expected = ALLOWED_EXPORT_OPTIONS[name]
        if not isinstance(value, expected):
            raise AssistanceValidationError(
                f"export option {name!r} must be {expected.__name__}")
        checked_options[name] = value

    index = payload.get("output_index")
    if index is not None:
        if isinstance(index, bool) or not isinstance(index, int):
            raise AssistanceValidationError("output_index must be an integer")
        if not 0 <= index < 8:
            raise AssistanceValidationError("output_index is out of range")

    adjustment = ExportAdjustment(
        summary=str(payload.get("summary") or "")[:MAX_SUMMARY_LENGTH],
        module_attributes=checked_attributes,
        export_options=checked_options,
        output_index=index,
    )
    if adjustment.is_empty:
        raise AssistanceValidationError(
            "the adjustment proposes no change DelegateDoctor can apply")
    return adjustment


def parse_adjustment_text(text: str) -> ExportAdjustment:
    """Parse a reply. Fenced JSON tolerated; prose and code are not."""
    stripped = (text or "").strip()
    if not stripped:
        raise AssistanceValidationError("the AI provider returned nothing")

    if stripped.startswith("```"):
        stripped = "\n".join(line for line in stripped.splitlines()
                             if not line.strip().startswith("```")).strip()
    if not stripped.startswith("{"):
        start, end = stripped.find("{"), stripped.rfind("}")
        if start == -1 or end <= start:
            raise AssistanceValidationError(
                "the AI provider did not return a JSON adjustment")
        stripped = stripped[start:end + 1]

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as error:
        raise AssistanceValidationError(
            f"the adjustment was not valid JSON: {error.msg}")
    return parse_adjustment(payload)


# --- the request ----------------------------------------------------------------


SYSTEM_PROMPT = """You are DelegateDoctor's export-assistance assistant.

The user has already supplied a valid DelegateDoctor model interface. Their
model factory and input factory are authoritative and must not be replaced.
Your job is only to propose the smallest safe adjustment that makes the
existing interface exportable with torch.export.

Return one JSON object and nothing else. Never return code, prose, imports,
file paths or URLs.
"""


def build_prompt(source_text: str, failure, model_function: str,
                 inputs_function: str, optional_functions=()) -> str:
    """The outbound message: the interface, the failure, and the source shown.

    Explicitly states that the interface is authoritative, because the generic
    preparation prompt's question - "which symbol is the model?" - is exactly
    the wrong one here and produced the contradictory rejection this path
    exists to remove.
    """
    optional = ("\n".join(f"    {name}()" for name in optional_functions)
                or "    (none)")

    return f"""The user supplied a valid DelegateDoctor model interface. It was
found, executed, and it built a model successfully.

Model factory:
    {model_function}()

Input factory:
    {inputs_function}()

Optional factories:
{optional}

torch.export then failed:

    stage:   {failure.stage}
    type:    {failure.exception_type or 'unknown'}
    message: {failure.message or '(none reported)'}

{failure.traceback_text}

Selected source:

{source_text}

Analyze the source and the failure, and propose the smallest safe change to
model construction or export configuration that would make this existing
interface exportable. Do NOT propose a different model symbol, a different
class, or a replacement for the interface functions - they are correct and
they already worked.

Return one JSON object:

{{
  "summary": "one short sentence",
  "module_attributes": {{"aux_logits": false}},
  "export_options": {{"strict": false}},
  "output_index": 0
}}

Every field is optional but at least one must be present.
  module_attributes  set an attribute on the model the factory already built.
                     Values must be numbers, booleans or null - never strings.
  export_options     only these keys: {', '.join(sorted(ALLOWED_EXPORT_OPTIONS))}.
  output_index       keep one output when the forward returns several.

Nothing else is accepted. Do not include reasoning."""


# --- the loop -------------------------------------------------------------------


ASSISTANCE_UNAVAILABLE = "AI export assistance is unavailable"
ASSISTANCE_NO_CHANGE = "AI export assistance proposed no valid change"
ASSISTANCE_STILL_FAILING = "the adjusted interface still did not export"


@dataclass
class AssistanceOutcome:
    """What assistance achieved, or precisely how far it got."""

    prepared: object = None
    adjustment: Optional[ExportAdjustment] = None
    attempts: list = field(default_factory=list)
    reason: str = ""

    @property
    def succeeded(self) -> bool:
        return self.prepared is not None

    def to_dict(self) -> dict:
        return {
            "succeeded": self.succeeded,
            "adjustment": (self.adjustment.to_dict()
                           if self.adjustment else None),
            "attempts": list(self.attempts),
            "reason": self.reason,
        }


def assist_export(model_path, workspace, failure, provider, source_text: str,
                  export, announce=print, verbose: bool = False,
                  max_attempts: int = MAX_EXPORT_ASSISTANCE_ATTEMPTS
                  ) -> AssistanceOutcome:
    """Ask for an adjustment, apply it, and retry the *same* interface.

    `export` is injected - it is `model_interface.prepare_from_interface` -
    so this module never acquires its own idea of what exporting means, and
    the retry runs in exactly the sanitized child the first attempt did.

    Bounded: at most `max_attempts` adjustments, each costing one request and
    one export. A third guess is rarely a better guess.
    """
    from .. import model_interface
    from .client import AIError, AIRequest

    outcome = AssistanceOutcome()
    previous = ""

    for attempt in range(1, max_attempts + 1):
        prompt = build_prompt(
            source_text, failure,
            model_interface.MODEL_FUNCTION, model_interface.INPUTS_FUNCTION,
            (model_interface.KWARGS_FUNCTION,
             model_interface.DYNAMIC_SHAPES_FUNCTION))
        if previous:
            prompt += f"\n\nYour previous adjustment was not usable:\n{previous}"

        try:
            response = provider.complete_structured(
                AIRequest(system=SYSTEM_PROMPT, user=prompt,
                          purpose="export-assistance"))
        except AIError as error:
            outcome.reason = f"{ASSISTANCE_UNAVAILABLE}: {error}"
            outcome.attempts.append({"attempt": attempt,
                                     "outcome": "provider-error"})
            return outcome

        try:
            adjustment = parse_adjustment_text(response.text)
        except AssistanceValidationError as error:
            previous = str(error)[:300]
            announce(f"    Adjustment {attempt}          rejected by validation")
            if verbose:
                announce(f"    Reason                  {previous}")
            outcome.attempts.append({"attempt": attempt, "outcome": "invalid",
                                     "detail": previous})
            continue

        announce(f"    Export assistance       {adjustment.describe()}")
        try:
            prepared = export(model_path, workspace, adjustment=adjustment,
                              quiet=True)
        except model_interface.ModelInterfaceError as error:
            retry_failure = getattr(error, "failure", None) or failure
            previous = retry_failure.summary()[:300]
            announce("    Export retry            FAILED")
            if verbose:
                announce(f"    Reason                  {previous}")
            outcome.attempts.append({"attempt": attempt, "outcome": "still-failing",
                                     "detail": previous})
            failure = retry_failure
            continue

        outcome.prepared = prepared
        outcome.adjustment = adjustment
        outcome.attempts.append({"attempt": attempt, "outcome": "exported",
                                 "detail": adjustment.describe()})
        return outcome

    outcome.reason = (ASSISTANCE_STILL_FAILING
                      if any(item["outcome"] == "still-failing"
                             for item in outcome.attempts)
                      else ASSISTANCE_NO_CHANGE)
    return outcome
