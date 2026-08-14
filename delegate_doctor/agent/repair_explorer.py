"""Ask the provider for a candidate repair, and hand it to the usual gates.

Reached only when everything deterministic has already been tried: the graph
lowered, the target profiled, a portable hotspot was actually measured, and no
catalog rule matched it. The catalog always wins - if DD-001 or DD-002 applies,
this module is never called and no request is made.

What comes back is a `RepairCandidatePlan`, not code. DelegateDoctor validates
it, applies it to a fresh copy of the pristine baseline, checks the graph, and
then lets the *existing* verification and benchmark gates decide. Nothing here
can accept a repair.

    candidate -> schema -> pristine copy -> apply -> lint -> lower
      -> (the ordinary host/device/benchmark gates)

Retries exist only for proposals that never became runnable. Once a candidate
reaches the real gates and fails one, exploration stops: a correctness or
latency failure is an answer, not a prompt for another guess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import graph_context, prompts
from .client import AIRequest, AIResponse
from .repair_applier import CandidateApplicationError, apply_candidate
from .repair_schema import (MAX_AI_REPAIR_OPERATIONS, CandidateValidationError,
                            RepairCandidatePlan, parse_candidate_text)

# Two proposals, and only for pre-gate failures. The bound belongs to
# DelegateDoctor: nothing in a provider response can change it.
MAX_AI_REPAIR_CANDIDATES = 2


@dataclass
class CandidateAttempt:
    """A sanitized record of one proposal, safe for artifacts and the report."""

    candidate_id: str
    outcome: str              # "invalid", "not-lowerable", "runnable"
    detail: str = ""
    summary: str = ""

    def to_dict(self) -> dict:
        return {"candidate": self.candidate_id, "outcome": self.outcome,
                "detail": self.detail[:300], "summary": self.summary[:300]}


@dataclass
class ExplorationResult:
    """What exploration produced. At most one runnable candidate."""

    program: object = None            # the rewritten ExportedProgram, or None
    plan: RepairCandidatePlan = None
    attempts: list = field(default_factory=list)

    # What the provider call itself resolved to. A transport failure, a
    # refusal or an empty response is recorded *here* - not as a candidate,
    # because no candidate existed to record.
    provider_result: object = None

    # The model answered and its answer was "no safe repair". A successful
    # response, and emphatically not an error.
    declined: bool = False

    @property
    def provider_succeeded(self) -> bool:
        return (self.provider_result is None
                or getattr(self.provider_result, "succeeded", False))

    @property
    def candidates_proposed(self) -> int:
        """Structured proposals the provider actually returned."""
        return len(self.attempts)

    @property
    def found_runnable(self) -> bool:
        return self.program is not None

    @property
    def runnable_candidates(self) -> list:
        """`[(program, plan), ...]` for the caller to gate, in order.

        At most one, because the candidate bound is per exploration and a
        proposal that reaches the real gates ends the round either way. It is
        a list so the pipeline treats "what this exploration produced" as a
        queue of ordinary candidates rather than a special case.
        """
        if not self.found_runnable:
            return []
        return [(self.program, self.plan)]

    @property
    def candidate_count(self) -> int:
        return len(self.attempts)

    def to_dict(self) -> dict:
        return {"candidates": [attempt.to_dict() for attempt in self.attempts],
                "runnable": self.found_runnable}


def build_request(context: dict, previous_failure: str = "") -> AIRequest:
    """The outbound payload: graph and measurement only, never source.

    The question is about the model, not about one operator. Asking "fix
    native_layer_norm.out" forces every answer to have the same boundaries as
    an ETDump event name, and the transformations worth finding often span
    several operators at once.
    """
    parts = [
        "DelegateDoctor measured this model on an Arm64 Android target.",
        "",
        json.dumps(context, indent=2, sort_keys=True),
        "",
        "Analyze this exported model and its measured execution. Propose one "
        "constrained rewrite that would reduce portable (non-delegated) "
        "execution while preserving semantics. The rewrite may involve "
        "several operators; it does not have to correspond to a single "
        "profiled operator, and it should not duplicate a repair listed under "
        "known_repairs.",
        "",
        _schema_reminder(),
    ]
    if previous_failure:
        parts += ["", "Your previous candidate was not usable:", previous_failure,
                  "", "Propose a different one."]
    return AIRequest(system=prompts.REPAIR_SYSTEM, user="\n".join(parts),
                    purpose="repair")


def _declines_to_propose(text: str) -> bool:
    """Did the model answer with an explicit "no safe repair"?

    A one-key JSON object, recognised before schema validation so that a
    deliberate decline is not reported as malformed output. This adds nothing
    to the repair vocabulary - it is how the model says it is not using it.
    """
    import json

    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = "\n".join(line for line in stripped.splitlines()
                             if not line.strip().startswith("```")).strip()
    if not stripped.startswith("{"):
        return False
    try:
        payload = json.loads(stripped)
    except Exception:
        return False
    return isinstance(payload, dict) and payload.get("no_repair") is True


def _schema_reminder() -> str:
    from .repair_schema import ALLOWED_ATEN_TARGETS, ALLOWED_OPERATIONS

    return (
        "Return a single JSON object and nothing else:\n"
        "\n"
        "{\n"
        '  "summary": "one sentence",\n'
        '  "anchor": "node_N (must be a node above)",\n'
        '  "operations": [\n'
        '    {"type": "insert_aten_call", "id": "new_1",\n'
        '     "target": "aten.reshape.default",\n'
        '     "args": [{"node": "node_N"}, [1, 2, 3]],\n'
        '     "before": "node_N"},\n'
        '    {"type": "replace_uses", "old": "node_N", "new": "new_1"}\n'
        "  ]\n"
        "}\n"
        "\n"
        f"Operation types: {', '.join(ALLOWED_OPERATIONS)}\n"
        f"At most {MAX_AI_REPAIR_OPERATIONS} operations.\n"
        f"Allowed targets: {', '.join(ALLOWED_ATEN_TARGETS)}\n"
        "\n"
        "Arguments may only be node references, integers, finite floats, "
        "booleans, null, or short lists of those. Never strings, code, paths "
        "or URLs.\n"
        "\n"
        "If no safe rewrite is expressible in this DSL, return exactly:\n"
        "\n"
        '{"no_repair": true, "reason": "one short sentence"}'
    )


def explore(provider, baseline_program, context: dict, known_nodes,
            lower, announce=print,
            max_candidates: int = MAX_AI_REPAIR_CANDIDATES,
            candidate_id_factory=None) -> ExplorationResult:
    """Ask for candidates until one becomes a runnable graph, or the bound is hit.

    `lower` is injected - it is the pipeline's own ExecuTorch lowering - so this
    module never acquires its own idea of what lowering means.

    The bound applies to *this hotspot*. A candidate that reaches the real
    gates and fails one ends exploration here, but says nothing about the next
    hotspot, which the repair loop is free to explore independently.

    `candidate_id_factory` lets the caller number candidates across a whole
    optimization run. Without it, numbering restarts at 001 for every hotspot,
    which would make `AI-CANDIDATE-002` ambiguous in a report.
    """
    result = ExplorationResult()
    previous_failure = ""

    for index in range(1, max_candidates + 1):
        candidate_id = (candidate_id_factory() if candidate_id_factory
                        else f"AI-CANDIDATE-{index:03d}")

        completion = provider.complete(build_request(context, previous_failure))
        if not completion.succeeded:
            # The provider did not return a usable payload. That is a fact
            # about the request, and inventing a candidate to hang it on is
            # what made "empty response" read as "the AI proposed something
            # unusable".
            result.provider_result = completion
            break
        result.provider_result = completion
        response = AIResponse(text=completion.text)

        # --- 0. did the model deliberately decline to propose anything? ---
        if _declines_to_propose(completion.text):
            result.declined = True
            break

        # --- 1. does it even parse and validate? --------------------------
        try:
            plan = parse_candidate_text(response.text, known_nodes, candidate_id)
        except CandidateValidationError as error:
            detail = str(error)[:300]
            announce(f"  {candidate_id}  rejected by validation")
            result.attempts.append(
                CandidateAttempt(candidate_id, "invalid", detail))
            previous_failure = detail
            continue

        # --- 2. apply to a *fresh* copy of the pristine baseline -----------
        try:
            rewritten = apply_candidate(baseline_program, plan)
        except (CandidateApplicationError, CandidateValidationError) as error:
            detail = str(error)[:300]
            announce(f"  {candidate_id}  did not produce a valid graph")
            result.attempts.append(CandidateAttempt(
                candidate_id, "invalid", detail, plan.summary))
            previous_failure = detail
            continue

        # --- 3. can ExecuTorch lower it? -----------------------------------
        try:
            lower(rewritten)
        except Exception as error:
            detail = f"{type(error).__name__}: {str(error)[:200]}"
            announce(f"  {candidate_id}  did not lower")
            result.attempts.append(CandidateAttempt(
                candidate_id, "not-lowerable", detail, plan.summary))
            previous_failure = detail
            continue

        announce(f"  {candidate_id}  runnable; entering the usual gates")
        result.attempts.append(CandidateAttempt(
            candidate_id, "runnable", "", plan.summary))
        result.program = rewritten
        result.plan = plan
        return result

    return result



def build_context(exported_program, hotspot_operator: str, profile, delegation,
                  executorch_version: str = "", node_name: str = "",
                  hotspot=None) -> tuple:
    """(context, known_node_ids). Reuses the Phase 6A neighbourhood extractor.

    `node_name` and `hotspot` identify *which* occurrence of the operator this
    request is about, which matters once the repair loop is working through
    several hotspots in one run.
    """
    neighbourhood = graph_context.build_neighbourhood(
        exported_program, hotspot_operator, node_name=node_name)
    context = graph_context.build_repair_context(
        neighbourhood, profile, delegation, executorch_version, hotspot=hotspot)
    known = [node.identifier for node in neighbourhood.nodes]
    return context, known
