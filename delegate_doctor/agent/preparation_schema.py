"""The only thing the model is allowed to say back about preparation.

The agent does not write the export adapter. It fills in a narrow form, every
field of which is validated here, and DelegateDoctor builds the adapter from
the validated result. That inversion is the whole security design: there is no
path from provider output to executed source, because provider output is never
source.

What a plan may contain:

    which symbol in the file to use, and whether it is a class or an instance
    literal constructor arguments (numbers, strings, booleans, small lists)
    a local checkpoint file name, if there is one
    input tensor specs - shape, dtype, generator - and nothing else

What it may not contain, and is rejected for containing:

    Python expressions, imports, code snippets
    shell commands
    URLs or absolute paths
    anything that is not a plain literal
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# Deliberately small. Anything not on this list is rejected rather than
# interpreted, so widening the surface is a conscious act.
ALLOWED_DTYPES = ("float32", "float16", "bfloat16", "int64", "int32", "bool",
                  "uint8", "int8")
ALLOWED_GENERATORS = ("randn", "zeros", "ones", "rand", "randint", "arange")

SYMBOL_KIND_CLASS = "class"
SYMBOL_KIND_INSTANCE = "existing_instance"
ALLOWED_SYMBOL_KINDS = (SYMBOL_KIND_CLASS, SYMBOL_KIND_INSTANCE)

MAX_TENSOR_RANK = 8
MAX_DIMENSION = 65_536
MAX_INPUTS = 8
MAX_CONSTRUCTOR_ARGUMENTS = 16
MAX_STRING_LENGTH = 128

# A valid Python identifier and nothing else: no dots, no calls, no subscripts.
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Shapes that indicate the model tried to smuggle code or a location into a
# field that is supposed to hold a plain literal.
_FORBIDDEN_IN_STRINGS = ("://", "..", "\n", "\r", "\x00", "`", "$(", "${",
                         "import ", "lambda", "eval(", "exec(", "os.", "sys.",
                         "subprocess", "__")


class PlanValidationError(ValueError):
    """The proposed plan was not accepted. It is never executed."""


@dataclass(frozen=True)
class TensorInputSpec:
    """One representative input, described only by controlled metadata."""

    shape: tuple
    dtype: str = "float32"
    generator: str = "randn"

    def describe(self) -> str:
        return f"{self.dtype} {list(self.shape)}"


@dataclass
class PreparationPlan:
    """A validated description of how to construct and export the model."""

    model_name: str
    symbol: str
    symbol_kind: str
    constructor_args: list = field(default_factory=list)
    constructor_kwargs: dict = field(default_factory=dict)
    checkpoint: str = ""
    positional_inputs: list = field(default_factory=list)
    keyword_inputs: dict = field(default_factory=dict)
    notes: str = ""
    confidence: str = ""
    missing_information: list = field(default_factory=list)

    @property
    def needs_user_input(self) -> bool:
        return bool(self.missing_information) and not self.positional_inputs

    def describe_inputs(self) -> str:
        parts = [spec.describe() for spec in self.positional_inputs]
        parts += [f"{name}={spec.describe()}"
                  for name, spec in sorted(self.keyword_inputs.items())]
        return ", ".join(parts)


# --- validation --------------------------------------------------------------


def _check_identifier(value, field_name: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.match(value):
        raise PlanValidationError(
            f"{field_name} must be a plain Python identifier, got {value!r}")
    return value


def _check_literal(value, field_name: str, depth: int = 0):
    """Only plain data. No expressions, no paths, no callables."""
    if depth > 2:
        raise PlanValidationError(f"{field_name} is nested too deeply")

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if abs(value) > 2 ** 31:
            raise PlanValidationError(f"{field_name} integer is out of range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PlanValidationError(
                f"{field_name} must be a finite number, got {value!r}")
        return value
    if isinstance(value, str):
        if len(value) > MAX_STRING_LENGTH:
            raise PlanValidationError(f"{field_name} string is too long")
        lowered = value.lower()
        for forbidden in _FORBIDDEN_IN_STRINGS:
            if forbidden in lowered:
                raise PlanValidationError(
                    f"{field_name} contains disallowed content ({forbidden!r}). "
                    f"Only plain literal values are accepted.")
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_CONSTRUCTOR_ARGUMENTS:
            raise PlanValidationError(f"{field_name} list is too long")
        return [_check_literal(item, field_name, depth + 1) for item in value]

    raise PlanValidationError(
        f"{field_name} must be a literal (number, string, bool, null or list), "
        f"got {type(value).__name__}")


def _parse_tensor_spec(raw, field_name: str) -> TensorInputSpec:
    if not isinstance(raw, dict):
        raise PlanValidationError(f"{field_name} must be an object")

    unknown = set(raw) - {"shape", "dtype", "generator"}
    if unknown:
        raise PlanValidationError(
            f"{field_name} has unknown field(s): {', '.join(sorted(unknown))}")

    shape = raw.get("shape")
    if not isinstance(shape, (list, tuple)) or not shape:
        raise PlanValidationError(f"{field_name}.shape must be a non-empty list")
    if len(shape) > MAX_TENSOR_RANK:
        raise PlanValidationError(
            f"{field_name}.shape has rank {len(shape)}, above the "
            f"{MAX_TENSOR_RANK} DelegateDoctor accepts")
    dimensions = []
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise PlanValidationError(
                f"{field_name}.shape must contain integers, got {dimension!r}")
        if dimension < 1 or dimension > MAX_DIMENSION:
            raise PlanValidationError(
                f"{field_name}.shape has an out-of-range dimension: {dimension}")
        dimensions.append(dimension)

    dtype = raw.get("dtype", "float32")
    if dtype not in ALLOWED_DTYPES:
        raise PlanValidationError(
            f"{field_name}.dtype {dtype!r} is not one of: "
            f"{', '.join(ALLOWED_DTYPES)}")

    generator = raw.get("generator", "randn")
    if generator not in ALLOWED_GENERATORS:
        raise PlanValidationError(
            f"{field_name}.generator {generator!r} is not one of: "
            f"{', '.join(ALLOWED_GENERATORS)}")

    return TensorInputSpec(shape=tuple(dimensions), dtype=dtype,
                           generator=generator)


def _check_checkpoint(value) -> str:
    """A bare local file name, or nothing.

    No directories and no URLs: DelegateDoctor resolves the file next to the
    model source, so a path is not only unnecessary but a way to reach
    somewhere it should not.
    """
    if value in (None, ""):
        return ""
    if not isinstance(value, str):
        raise PlanValidationError("checkpoint must be a file name or null")
    if len(value) > MAX_STRING_LENGTH:
        raise PlanValidationError("checkpoint name is too long")
    if "/" in value or "\\" in value or "://" in value or value.startswith("~"):
        raise PlanValidationError(
            f"checkpoint must be a bare file name beside the model source, "
            f"got {value!r}")
    if ".." in value:
        raise PlanValidationError("checkpoint may not traverse directories")
    return value


KNOWN_FIELDS = {
    "model_name", "symbol", "symbol_kind", "constructor_args",
    "constructor_kwargs", "checkpoint", "positional_inputs", "keyword_inputs",
    "notes", "confidence", "missing_information",
}


def parse_plan(payload: dict) -> PreparationPlan:
    """Validate a proposed plan. Raises rather than returning something unsafe."""
    if not isinstance(payload, dict):
        raise PlanValidationError("The plan must be a JSON object.")

    unknown = set(payload) - KNOWN_FIELDS
    if unknown:
        raise PlanValidationError(
            f"The plan has unknown field(s): {', '.join(sorted(unknown))}")

    missing = payload.get("missing_information") or []
    if not isinstance(missing, list):
        raise PlanValidationError("missing_information must be a list")
    missing = [str(item)[:200] for item in missing]

    symbol = payload.get("symbol")
    if symbol is None and missing:
        # A plan that honestly reports it could not decide. Valid, and handled
        # by asking the user rather than by inventing a symbol.
        return PreparationPlan(
            model_name=str(payload.get("model_name") or "")[:120],
            symbol="", symbol_kind=SYMBOL_KIND_CLASS,
            notes=str(payload.get("notes") or "")[:500],
            confidence=str(payload.get("confidence") or "")[:40],
            missing_information=missing,
        )

    _check_identifier(symbol, "symbol")

    symbol_kind = payload.get("symbol_kind", SYMBOL_KIND_CLASS)
    if symbol_kind not in ALLOWED_SYMBOL_KINDS:
        raise PlanValidationError(
            f"symbol_kind must be one of {ALLOWED_SYMBOL_KINDS}, "
            f"got {symbol_kind!r}")

    constructor_args = payload.get("constructor_args") or []
    if not isinstance(constructor_args, list):
        raise PlanValidationError("constructor_args must be a list")
    if len(constructor_args) > MAX_CONSTRUCTOR_ARGUMENTS:
        raise PlanValidationError("constructor_args has too many entries")
    constructor_args = [_check_literal(value, "constructor_args")
                        for value in constructor_args]

    constructor_kwargs = payload.get("constructor_kwargs") or {}
    if not isinstance(constructor_kwargs, dict):
        raise PlanValidationError("constructor_kwargs must be an object")
    if len(constructor_kwargs) > MAX_CONSTRUCTOR_ARGUMENTS:
        raise PlanValidationError("constructor_kwargs has too many entries")
    checked_kwargs = {}
    for name, value in constructor_kwargs.items():
        _check_identifier(name, "constructor_kwargs key")
        checked_kwargs[name] = _check_literal(value, f"constructor_kwargs.{name}")

    positional = payload.get("positional_inputs") or []
    if not isinstance(positional, list):
        raise PlanValidationError("positional_inputs must be a list")
    if len(positional) > MAX_INPUTS:
        raise PlanValidationError("positional_inputs has too many entries")
    positional_specs = [_parse_tensor_spec(raw, f"positional_inputs[{index}]")
                        for index, raw in enumerate(positional)]

    keyword = payload.get("keyword_inputs") or {}
    if not isinstance(keyword, dict):
        raise PlanValidationError("keyword_inputs must be an object")
    if len(keyword) > MAX_INPUTS:
        raise PlanValidationError("keyword_inputs has too many entries")
    keyword_specs = {}
    for name, raw in keyword.items():
        _check_identifier(name, "keyword_inputs key")
        keyword_specs[name] = _parse_tensor_spec(raw, f"keyword_inputs.{name}")

    if not positional_specs and not keyword_specs and not missing:
        raise PlanValidationError(
            "The plan supplies no inputs and reports nothing missing. "
            "DelegateDoctor will not guess an input shape.")

    return PreparationPlan(
        model_name=str(payload.get("model_name") or symbol)[:120],
        symbol=symbol,
        symbol_kind=symbol_kind,
        constructor_args=constructor_args,
        constructor_kwargs=checked_kwargs,
        checkpoint=_check_checkpoint(payload.get("checkpoint")),
        positional_inputs=positional_specs,
        keyword_inputs=keyword_specs,
        notes=str(payload.get("notes") or "")[:500],
        confidence=str(payload.get("confidence") or "")[:40],
        missing_information=missing,
    )


def parse_plan_text(text: str) -> PreparationPlan:
    """Parse the model's reply. Fenced JSON is tolerated; prose is not."""
    import json

    stripped = (text or "").strip()
    if not stripped:
        raise PlanValidationError("The AI provider returned an empty plan.")

    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines()
                 if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()

    if not stripped.startswith("{"):
        start, end = stripped.find("{"), stripped.rfind("}")
        if start == -1 or end <= start:
            raise PlanValidationError(
                "The AI provider did not return a JSON preparation plan.")
        stripped = stripped[start:end + 1]

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as error:
        raise PlanValidationError(f"The plan was not valid JSON: {error.msg}")

    return parse_plan(payload)
