"""The only thing an AI may say about a repair: a validated rewrite plan.

There is no path from provider output to executed code. The agent does not
return Python, and nothing here compiles, evaluates or imports anything it is
given. It returns a short list of operations drawn from a fixed vocabulary,
naming operators from a fixed allowlist, with arguments that must be plain
literals or references to nodes that already exist.

    {"summary": "...",
     "anchor": "node_17",
     "operations": [
       {"type": "insert_aten_call", "id": "new_1",
        "target": "aten.reshape.default",
        "args": [{"node": "node_12"}, [1, 224, 224, 32]],
        "before": "node_17"},
       {"type": "replace_uses", "old": "node_17", "new": "new_1"},
       {"type": "erase_node", "node": "node_17"}
     ]}

Everything is bounded: how many operations, how many new nodes, how large a
literal may be. A plan that fails any check is never applied, and the graph is
never touched to find out.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# --- the operator allowlist --------------------------------------------------
#
# Structural and elementwise operations useful for making a graph
# backend-friendly. Deliberately short: an operator is added here when there is
# a reason, not because it exists. Nothing that touches files, processes, the
# network, or custom/user-defined operators can appear.

ALLOWED_ATEN_TARGETS = (
    # layout and shape
    "aten.reshape.default",
    "aten.view.default",
    "aten.permute.default",
    "aten.transpose.int",
    "aten.squeeze.dim",
    "aten.unsqueeze.default",
    "aten.flatten.using_ints",
    "aten.contiguous.default",
    "aten.clone.default",
    "aten.expand.default",
    "aten.slice.Tensor",
    "aten.cat.default",
    # elementwise
    "aten.add.Tensor",
    "aten.sub.Tensor",
    "aten.mul.Tensor",
    "aten.div.Tensor",
    "aten.relu.default",
    "aten.sigmoid.default",
    "aten.tanh.default",
    # reductions and normalization
    "aten.softmax.int",
    "aten._softmax.default",
    "aten.sum.dim_IntList",
    "aten.mean.dim",
    "aten.max.dim",
)

OPERATION_INSERT = "insert_aten_call"
OPERATION_REPLACE_USES = "replace_uses"
OPERATION_REPLACE_ARGUMENT = "replace_argument"
OPERATION_ERASE = "erase_node"

ALLOWED_OPERATIONS = (OPERATION_INSERT, OPERATION_REPLACE_USES,
                      OPERATION_REPLACE_ARGUMENT, OPERATION_ERASE)

# Bounds. A giant LLM-authored rewrite is not something to apply and find out.
MAX_AI_REPAIR_OPERATIONS = 12
MAX_NEW_NODES = 6
MAX_ARGUMENTS = 8
MAX_LITERAL_ITEMS = 16
MAX_SUMMARY_LENGTH = 400

NODE_REFERENCE = re.compile(r"^(node|new)_[A-Za-z0-9_]{1,32}$")


class CandidateValidationError(ValueError):
    """The proposed candidate was rejected. It is never applied."""


@dataclass(frozen=True)
class NodeReference:
    """A reference to an existing graph node or one this plan creates."""

    identifier: str


@dataclass
class Operation:
    kind: str
    payload: dict = field(default_factory=dict)


@dataclass
class RepairCandidatePlan:
    """A validated rewrite plan. Still only a proposal."""

    summary: str
    anchor: str
    operations: list = field(default_factory=list)
    candidate_id: str = "AI-CANDIDATE-001"

    @property
    def new_node_ids(self) -> list:
        return [operation.payload["id"] for operation in self.operations
                if operation.kind == OPERATION_INSERT]

    def describe(self) -> str:
        kinds = ", ".join(operation.kind for operation in self.operations)
        return f"{self.candidate_id}: {len(self.operations)} operation(s) [{kinds}]"


# --- validation ---------------------------------------------------------------


def _check_reference(value, field_name: str) -> str:
    if not isinstance(value, str) or not NODE_REFERENCE.match(value):
        raise CandidateValidationError(
            f"{field_name} must be a node reference like 'node_12' or 'new_1', "
            f"got {value!r}")
    return value


def _check_argument(value, field_name: str, depth: int = 0):
    """A node reference, or a plain literal. Never anything executable."""
    if depth > 2:
        raise CandidateValidationError(f"{field_name} is nested too deeply")

    if isinstance(value, dict):
        if set(value) != {"node"}:
            raise CandidateValidationError(
                f"{field_name} object must be exactly {{'node': '...'}}, "
                f"got keys {sorted(value)}")
        return NodeReference(_check_reference(value["node"], field_name))

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if abs(value) > 2 ** 31:
            raise CandidateValidationError(f"{field_name} integer out of range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CandidateValidationError(
                f"{field_name} must be finite, got {value!r}")
        return value
    if isinstance(value, str):
        # Strings are where code, paths and URLs would hide. No ATen call in
        # the allowlist needs one, so none is accepted.
        raise CandidateValidationError(
            f"{field_name} may not be a string. Arguments must be node "
            f"references, numbers, booleans, null, or lists of those.")
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_LITERAL_ITEMS:
            raise CandidateValidationError(f"{field_name} list is too long")
        return [_check_argument(item, field_name, depth + 1) for item in value]

    raise CandidateValidationError(
        f"{field_name} has unsupported type {type(value).__name__}")


def _check_insert(payload: dict, known: set, created: set) -> Operation:
    unknown = set(payload) - {"type", "id", "target", "args", "before"}
    if unknown:
        raise CandidateValidationError(
            f"insert_aten_call has unknown field(s): {', '.join(sorted(unknown))}")

    identifier = _check_reference(payload.get("id"), "insert_aten_call.id")
    if not identifier.startswith("new_"):
        raise CandidateValidationError(
            f"a newly created node must be named new_*, got {identifier!r}")
    if identifier in created or identifier in known:
        raise CandidateValidationError(f"duplicate node id {identifier!r}")

    target = payload.get("target")
    if target not in ALLOWED_ATEN_TARGETS:
        raise CandidateValidationError(
            f"target {target!r} is not on DelegateDoctor's allowlist. "
            f"Allowed: {', '.join(ALLOWED_ATEN_TARGETS[:6])}, ...")

    raw_args = payload.get("args") or []
    if not isinstance(raw_args, list):
        raise CandidateValidationError("insert_aten_call.args must be a list")
    if len(raw_args) > MAX_ARGUMENTS:
        raise CandidateValidationError("insert_aten_call.args has too many entries")
    args = [_check_argument(value, f"args[{index}]")
            for index, value in enumerate(raw_args)]

    for argument in args:
        if isinstance(argument, NodeReference):
            if argument.identifier not in known and argument.identifier not in created:
                raise CandidateValidationError(
                    f"args references unknown node {argument.identifier!r}")

    before = payload.get("before")
    if before is not None:
        _check_reference(before, "insert_aten_call.before")
        if before not in known:
            raise CandidateValidationError(
                f"before references unknown node {before!r}")

    created.add(identifier)
    return Operation(OPERATION_INSERT, {
        "id": identifier, "target": target, "args": args, "before": before,
    })


def _check_replace_uses(payload: dict, known: set, created: set) -> Operation:
    unknown = set(payload) - {"type", "old", "new"}
    if unknown:
        raise CandidateValidationError(
            f"replace_uses has unknown field(s): {', '.join(sorted(unknown))}")
    old = _check_reference(payload.get("old"), "replace_uses.old")
    new = _check_reference(payload.get("new"), "replace_uses.new")
    if old not in known:
        raise CandidateValidationError(f"replace_uses.old is unknown: {old!r}")
    if new not in known and new not in created:
        raise CandidateValidationError(f"replace_uses.new is unknown: {new!r}")
    if old == new:
        raise CandidateValidationError("replace_uses.old and .new are the same")
    return Operation(OPERATION_REPLACE_USES, {"old": old, "new": new})


def _check_replace_argument(payload: dict, known: set, created: set) -> Operation:
    unknown = set(payload) - {"type", "node", "index", "value"}
    if unknown:
        raise CandidateValidationError(
            f"replace_argument has unknown field(s): {', '.join(sorted(unknown))}")
    node = _check_reference(payload.get("node"), "replace_argument.node")
    if node not in known and node not in created:
        raise CandidateValidationError(f"replace_argument.node is unknown: {node!r}")
    index = payload.get("index")
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < MAX_ARGUMENTS:
        raise CandidateValidationError(
            f"replace_argument.index must be an integer in 0..{MAX_ARGUMENTS - 1}")
    value = _check_argument(payload.get("value"), "replace_argument.value")
    return Operation(OPERATION_REPLACE_ARGUMENT,
                     {"node": node, "index": index, "value": value})


def _check_erase(payload: dict, known: set, created: set) -> Operation:
    unknown = set(payload) - {"type", "node"}
    if unknown:
        raise CandidateValidationError(
            f"erase_node has unknown field(s): {', '.join(sorted(unknown))}")
    node = _check_reference(payload.get("node"), "erase_node.node")
    if node not in known and node not in created:
        raise CandidateValidationError(f"erase_node.node is unknown: {node!r}")
    return Operation(OPERATION_ERASE, {"node": node})


_CHECKERS = {
    OPERATION_INSERT: _check_insert,
    OPERATION_REPLACE_USES: _check_replace_uses,
    OPERATION_REPLACE_ARGUMENT: _check_replace_argument,
    OPERATION_ERASE: _check_erase,
}


def parse_candidate(payload: dict, known_nodes,
                    candidate_id: str = "AI-CANDIDATE-001") -> RepairCandidatePlan:
    """Validate a proposed candidate against the graph it claims to rewrite."""
    if not isinstance(payload, dict):
        raise CandidateValidationError("The candidate must be a JSON object.")

    unknown = set(payload) - {"summary", "anchor", "operations", "notes"}
    if unknown:
        raise CandidateValidationError(
            f"The candidate has unknown field(s): {', '.join(sorted(unknown))}")

    known = set(known_nodes)
    anchor = _check_reference(payload.get("anchor"), "anchor")
    if anchor not in known:
        raise CandidateValidationError(
            f"anchor {anchor!r} is not a node in the supplied neighbourhood")

    operations = payload.get("operations")
    if not isinstance(operations, list) or not operations:
        raise CandidateValidationError("operations must be a non-empty list")
    if len(operations) > MAX_AI_REPAIR_OPERATIONS:
        raise CandidateValidationError(
            f"the candidate has {len(operations)} operations, above the "
            f"{MAX_AI_REPAIR_OPERATIONS} DelegateDoctor will apply")

    created: set = set()
    checked = []
    for index, raw in enumerate(operations):
        if not isinstance(raw, dict):
            raise CandidateValidationError(f"operations[{index}] must be an object")
        kind = raw.get("type")
        if kind not in ALLOWED_OPERATIONS:
            raise CandidateValidationError(
                f"operations[{index}] has unknown type {kind!r}. "
                f"Allowed: {', '.join(ALLOWED_OPERATIONS)}")
        checked.append(_CHECKERS[kind](raw, known, created))

    if len(created) > MAX_NEW_NODES:
        raise CandidateValidationError(
            f"the candidate creates {len(created)} nodes, above the "
            f"{MAX_NEW_NODES} DelegateDoctor will add")

    return RepairCandidatePlan(
        summary=str(payload.get("summary") or "")[:MAX_SUMMARY_LENGTH],
        anchor=anchor,
        operations=checked,
        candidate_id=candidate_id,
    )


def parse_candidate_text(text: str, known_nodes,
                         candidate_id: str = "AI-CANDIDATE-001") -> RepairCandidatePlan:
    """Parse a reply. Fenced JSON tolerated; prose and code are not."""
    import json

    stripped = (text or "").strip()
    if not stripped:
        raise CandidateValidationError("The AI provider returned nothing.")

    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines()
                 if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()

    if not stripped.startswith("{"):
        start, end = stripped.find("{"), stripped.rfind("}")
        if start == -1 or end <= start:
            raise CandidateValidationError(
                "The AI provider did not return a JSON repair candidate.")
        stripped = stripped[start:end + 1]

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as error:
        raise CandidateValidationError(
            f"The candidate was not valid JSON: {error.msg}")

    return parse_candidate(payload, known_nodes, candidate_id)


# --- reusing a validated plan at an equivalent site -----------------------------


def rebase_plan(plan: RepairCandidatePlan, offset: int,
                known_nodes) -> RepairCandidatePlan:
    """The same plan, pointed at a structurally equivalent site.

    `node_N` identifiers are absolute positions in the graph, so applying a
    plan authored for one site at another is a matter of shifting every
    existing-node reference by the distance between them. Generated `new_*`
    ids are left alone: they name nodes the plan creates, not nodes it found.

    Only ever called for sites that share a structural signature, which is what
    makes the shifted positions refer to structurally equivalent nodes. The
    result is re-validated against the new site's known set, so a shift that
    lands outside the graph is rejected here rather than discovered halfway
    through a rewrite - and the reused plan still faces every gate the original
    did.
    """
    known = set(known_nodes)

    def shift(identifier: str) -> str:
        if not identifier.startswith("node_"):
            return identifier
        try:
            position = int(identifier[len("node_"):])
        except ValueError:
            raise CandidateValidationError(
                f"cannot rebase malformed reference {identifier!r}")
        moved = position + offset
        if moved < 0:
            raise CandidateValidationError(
                f"rebasing {identifier!r} by {offset} leaves the graph")
        return f"node_{moved}"

    def shift_value(value):
        if isinstance(value, NodeReference):
            return NodeReference(shift(value.identifier))
        if isinstance(value, list):
            return [shift_value(item) for item in value]
        return value

    operations = []
    for operation in plan.operations:
        payload = {}
        for field_name, value in operation.payload.items():
            if field_name in ("old", "new", "before", "node"):
                payload[field_name] = shift_value(
                    NodeReference(value) if isinstance(value, str) else value)
                if isinstance(payload[field_name], NodeReference):
                    payload[field_name] = payload[field_name].identifier
            elif field_name == "args":
                payload[field_name] = [shift_value(item) for item in value]
            else:
                payload[field_name] = value
        operations.append(Operation(kind=operation.kind, payload=payload))

    rebased = RepairCandidatePlan(
        summary=plan.summary,
        anchor=shift(plan.anchor),
        operations=operations,
        candidate_id=plan.candidate_id,
    )
    _assert_references_exist(rebased, known)
    return rebased


def _assert_references_exist(plan: RepairCandidatePlan, known: set) -> None:
    """Every existing-node reference must be a node in the new neighbourhood."""
    created = {operation.payload["id"] for operation in plan.operations
               if operation.kind == OPERATION_INSERT}

    def check(identifier: str, where: str) -> None:
        if identifier.startswith("new_"):
            if identifier not in created:
                raise CandidateValidationError(
                    f"{where} references unknown generated node {identifier!r}")
            return
        if identifier not in known:
            raise CandidateValidationError(
                f"{where} references {identifier!r}, which is not in this "
                f"site's neighbourhood")

    check(plan.anchor, "anchor")
    for operation in plan.operations:
        for field_name in ("old", "new", "before", "node"):
            value = operation.payload.get(field_name)
            if isinstance(value, str):
                check(value, f"{operation.kind}.{field_name}")
        for argument in operation.payload.get("args", []) or []:
            if isinstance(argument, NodeReference):
                check(argument.identifier, f"{operation.kind}.args")
