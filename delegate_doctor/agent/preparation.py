"""Turn a user's `model.py` into an ExportedProgram, with `torch.export` judging.

The loop is short and the authority is not the model:

    inspect locally -> ask consent -> propose a plan -> validate it
      -> build an adapter -> run it in a child process -> torch.export decides

A plan the agent is confident about but that `torch.export` rejects is a failed
plan. A plan the agent is unsure about that exports cleanly is a good one. At
most `MAX_PREPARATION_ATTEMPTS` rounds, each fed only a sanitized summary of
what went wrong.

The child process gets `sanitized_child_environment()`: the user's own model
code runs, but it never inherits the AI key or any cloud credential. If a model
needs a secret to construct itself, DelegateDoctor says so rather than
forwarding one.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import adapter_builder, consent, prompts, source_inspection
from .client import AIError, AIRequest
from .preparation_schema import PlanValidationError, PreparationPlan, parse_plan_text
from .privacy import redact, redact_home_paths, sanitized_child_environment

MAX_PREPARATION_ATTEMPTS = 3
EXPORT_TIMEOUT_SECONDS = 600

# How much of a child-process failure is worth sending back. Enough to name the
# problem, never a whole traceback with paths in it.
MAX_FEEDBACK_CHARACTERS = 800


def model_spec_from_outcome(outcome, name: str = "") -> object:
    """Load what preparation produced into the pipeline's ModelSpec.

    Deliberately not routed through `pt2_input.load_model_spec`: that is the
    *artifact* contract, which is narrower on purpose (a flat tuple of fp32
    tensors). Preparation may legitimately produce keyword inputs, and
    `ModelSpec` already carries them, so this converges on the same object one
    step earlier rather than widening the artifact rules to fit.
    """
    import torch

    from ..export_model import ModelSpec

    program = torch.export.load(str(outcome.exported_program_path))
    args, kwargs = torch.load(str(outcome.inputs_path), map_location="cpu",
                              weights_only=True)
    return ModelSpec(
        name=name or outcome.plan.model_name or "PyTorch Model",
        exported_program=program,
        example_args=tuple(args),
        example_kwargs=dict(kwargs or {}),
        description=f"prepared from {outcome.files_sent[0]}"
                    if outcome.files_sent else "prepared model",
    )


class PreparationError(RuntimeError):
    """Preparation could not produce an ExportedProgram."""


class PreparationNeedsInput(PreparationError):
    """The agent reported it cannot decide something, and did not guess."""


@dataclass
class PreparationOutcome:
    """What preparation produced, for the pipeline and the report."""

    exported_program_path: Path
    inputs_path: Path
    plan: PreparationPlan
    attempts: int = 1
    summary: str = ""
    files_sent: list = field(default_factory=list)


def _sanitize_child_output(text: str) -> str:
    """A short, path-free, secret-free description of what the child reported."""
    cleaned = redact_home_paths(redact(text or ""))
    # Drop the absolute file names pytest-style tracebacks carry.
    cleaned = re.sub(r'File "[^"]*[/\\]([^"/\\]+)"', r'File "\1"', cleaned)
    return cleaned.strip()[:MAX_FEEDBACK_CHARACTERS]


def run_adapter(adapter_path: Path, working_dir: Path,
                timeout: int = EXPORT_TIMEOUT_SECONDS) -> tuple:
    """Execute the adapter in a sanitized child process. Returns (ok, output).

    The user's model is *their* code and they chose to point at it, but it
    still does not run inside DelegateDoctor's process, and it still does not
    get an environment containing anybody's credentials.
    """
    try:
        completed = subprocess.run(
            [sys.executable, str(adapter_path)],
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=sanitized_child_environment(),
        )
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT: preparation did not finish within {timeout}s"

    output = (completed.stdout or "") + (completed.stderr or "")
    ok = (completed.returncode == 0
          and adapter_builder.SUCCESS_MARKER in (completed.stdout or ""))
    return ok, output


def _explain_child_failure(output: str) -> str:
    """Turn a child failure into something a user can act on."""
    cleaned = _sanitize_child_output(output)

    missing = re.search(r"MISSING_DEPENDENCY:(\S+)", cleaned)
    if missing:
        name = missing.group(1).strip().strip("'\"")
        return (
            f"MISSING DEPENDENCY: {name}\n"
            f"\n"
            f"Your model needs {name}, which is not installed in this\n"
            f"environment. DelegateDoctor does not install packages.\n"
            f"\n"
            f"    python -m pip install {name}\n"
            f"\n"
            f"then retry."
        )
    return cleaned


def prepare_model(
    model_path: Path,
    provider,
    interactive: bool = True,
    allow_source: bool = False,
    announce=print,
    prompt=input,
    max_attempts: int = MAX_PREPARATION_ATTEMPTS,
    work_dir: Path = None,
) -> PreparationOutcome:
    """Prepare `model_path` for torch.export, asking before sending anything."""
    model_path = Path(model_path).resolve()

    # --- 1. everything that can be learned locally, learned locally --------
    facts = source_inspection.inspect_source(model_path)
    announce(source_inspection.summarize_for_console(facts))

    outbound_source = source_inspection.prepare_source_for_transmission(facts)
    safe, reason = source_inspection.transmission_is_safe(outbound_source)
    if not safe:
        raise PreparationError(reason)

    # --- 2. consent, before a single character leaves the machine ----------
    decision = consent.request_source_consent(
        [model_path], interactive=interactive, preapproved=allow_source,
        announce=announce, prompt=prompt)
    if not decision.granted:
        raise PreparationError(decision.reason)

    files_sent = [model_path.name]

    # --- 3. the bounded loop, with torch.export as the judge ---------------
    module_name = model_path.stem
    workspace = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(
        prefix="delegate_doctor_preparation_"))
    workspace.mkdir(parents=True, exist_ok=True)

    facts_summary = source_inspection.summarize_for_console(facts)
    previous_failure = ""
    last_error = ""
    plan = None

    for attempt in range(1, max_attempts + 1):
        request = AIRequest(
            system=prompts.PREPARATION_SYSTEM,
            user=prompts.preparation_user_message(
                outbound_source, model_path.name, facts_summary,
                previous_failure),
            purpose="preparation",
        )
        try:
            response = provider.complete_structured(request)
        except AIError as error:
            raise PreparationError(str(error))

        try:
            plan = parse_plan_text(response.text)
        except PlanValidationError as error:
            last_error = str(error)
            previous_failure = (
                f"Your previous answer was rejected: {last_error}\n"
                f"Return only the JSON form, with literal values.")
            continue

        if plan.needs_user_input:
            raise PreparationNeedsInput(_missing_information_message(plan))

        announce(adapter_builder.summarize_plan(plan))

        adapter_path = workspace / "prepare_model.py"
        output_path = workspace / "prepared_model.pt2"
        adapter_path.write_text(adapter_builder.build_adapter_source(
            plan, module_name, model_path.parent, output_path))

        ok, output = run_adapter(adapter_path, model_path.parent)

        # The adapter is DelegateDoctor's, not the user's, and it has served
        # its purpose. Remove it rather than leaving generated code around.
        adapter_path.unlink(missing_ok=True)

        if ok and output_path.is_file():
            announce("PyTorch export          PASS")
            return PreparationOutcome(
                exported_program_path=output_path,
                inputs_path=output_path.with_suffix(".inputs.pt"),
                plan=plan,
                attempts=attempt,
                summary=adapter_builder.summarize_plan(plan),
                files_sent=files_sent,
            )

        explanation = _explain_child_failure(output)
        if "MISSING DEPENDENCY" in explanation:
            # Another attempt cannot help: nothing the agent proposes will
            # install a package, and DelegateDoctor will not either.
            raise PreparationError(explanation)

        last_error = explanation
        previous_failure = explanation

    raise PreparationError(
        f"PYTORCH EXPORT FAILED\n"
        f"\n"
        f"DelegateDoctor prepared an adapter for {model_path.name}, but "
        f"torch.export\n"
        f"rejected the model on {max_attempts} attempts.\n"
        f"\n"
        f"The model has not entered the DelegateDoctor analysis pipeline.\n"
        f"\n"
        f"Last failure:\n{last_error}\n"
        f"\n"
        f"Export the model yourself to see the full error:\n"
        f"\n"
        f"    exported = torch.export.export(model.eval(), example_inputs)"
    )


def _missing_information_message(plan: PreparationPlan) -> str:
    listed = "\n".join(f"  {item}" for item in plan.missing_information)
    return (
        f"AI PREPARATION NEEDS INPUT\n"
        f"\n"
        f"DelegateDoctor could not determine everything it needs, and will not\n"
        f"guess:\n"
        f"\n"
        f"{listed}\n"
        f"\n"
        f"Construct the model in Python with the values you know are right,\n"
        f"and analyze it directly - no preparation and no AI:\n"
        f"\n"
        f"    from delegate_doctor import optimize\n"
        f"    result = optimize(model.eval(), args=(example_input,))"
    )
