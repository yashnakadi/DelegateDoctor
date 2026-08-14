"""Apply a validated candidate to a *copy* of the graph. DelegateDoctor does it.

The interpreter below is the only thing that ever acts on a candidate, and it
understands exactly four operations. It resolves operator targets through the
allowlist in `repair_schema` - by table lookup, never by importing a name the
agent supplied - so an operator that is not on the list cannot be reached even
if validation somehow let it through.

The pristine baseline is never touched. Every candidate starts from its own
deep copy, so a failed attempt cannot contaminate the next one, and the
reference the correctness gate compares against stays exactly what the model
originally meant.
"""

from __future__ import annotations

import copy

from .repair_schema import (ALLOWED_ATEN_TARGETS, OPERATION_ERASE,
                            OPERATION_INSERT, OPERATION_REPLACE_ARGUMENT,
                            OPERATION_REPLACE_USES, CandidateValidationError,
                            NodeReference, RepairCandidatePlan)


class CandidateApplicationError(RuntimeError):
    """The candidate could not be applied. The graph copy is discarded."""


def _resolve_target(name: str):
    """Turn an allowlisted name into the actual ATen overload.

    Membership is checked first and the lookup walks a fixed attribute path, so
    this cannot be used to reach an arbitrary callable.
    """
    import torch

    if name not in ALLOWED_ATEN_TARGETS:
        raise CandidateValidationError(f"target {name!r} is not allowlisted")

    parts = name.split(".")
    if len(parts) != 3 or parts[0] != "aten":
        raise CandidateValidationError(f"malformed target {name!r}")

    _, operator, overload = parts
    namespace = getattr(torch.ops, "aten")
    packet = getattr(namespace, operator, None)
    if packet is None:
        raise CandidateValidationError(f"unknown ATen operator {name!r}")
    resolved = getattr(packet, overload, None)
    if resolved is None:
        raise CandidateValidationError(f"unknown ATen overload {name!r}")
    return resolved


def _identifier_map(exported_program) -> dict:
    """The same node_N naming the neighbourhood used, rebuilt on a fresh copy."""
    return {f"node_{index}": node
            for index, node in enumerate(exported_program.graph.nodes)}


def _materialize(value, nodes: dict):
    """Turn validated arguments into real graph arguments."""
    if isinstance(value, NodeReference):
        node = nodes.get(value.identifier)
        if node is None:
            raise CandidateApplicationError(
                f"candidate references {value.identifier}, which is not in the graph")
        return node
    if isinstance(value, list):
        return [_materialize(item, nodes) for item in value]
    return value


def _protected(node) -> str:
    """Why this node must not be erased, or '' if it may be."""
    op = getattr(node, "op", "")
    if op == "placeholder":
        return "it is a graph input"
    if op == "output":
        return "it is the graph output"
    if op == "get_attr":
        return "it is a parameter or buffer reference"
    if list(getattr(node, "users", {})):
        return "it still has users"
    return ""


def apply_candidate(baseline_program, plan: RepairCandidatePlan):
    """Return a new ExportedProgram with the candidate applied.

    `baseline_program` is never modified - it is deep-copied first, exactly as
    the deterministic rules do, so the pristine graph survives every attempt.
    """
    working = copy.deepcopy(baseline_program)
    graph = working.graph
    nodes = _identifier_map(working)

    for operation in plan.operations:
        payload = operation.payload

        if operation.kind == OPERATION_INSERT:
            target = _resolve_target(payload["target"])
            arguments = tuple(_materialize(value, nodes)
                              for value in payload["args"])
            anchor = nodes.get(payload["before"]) if payload["before"] else None
            try:
                if anchor is not None:
                    with graph.inserting_before(anchor):
                        created = graph.call_function(target, arguments)
                else:
                    created = graph.call_function(target, arguments)
            except Exception as error:
                raise CandidateApplicationError(
                    f"could not insert {payload['target']}: "
                    f"{type(error).__name__}: {str(error)[:200]}")
            nodes[payload["id"]] = created

        elif operation.kind == OPERATION_REPLACE_USES:
            old = nodes.get(payload["old"])
            new = nodes.get(payload["new"])
            if old is None or new is None:
                raise CandidateApplicationError(
                    "replace_uses referenced a node that does not exist")
            old.replace_all_uses_with(new)

        elif operation.kind == OPERATION_REPLACE_ARGUMENT:
            node = nodes.get(payload["node"])
            if node is None:
                raise CandidateApplicationError(
                    "replace_argument referenced a node that does not exist")
            arguments = list(node.args)
            index = payload["index"]
            if index >= len(arguments):
                raise CandidateApplicationError(
                    f"replace_argument index {index} is out of range for "
                    f"{payload['node']}")
            arguments[index] = _materialize(payload["value"], nodes)
            node.args = tuple(arguments)

        elif operation.kind == OPERATION_ERASE:
            node = nodes.get(payload["node"])
            if node is None:
                raise CandidateApplicationError(
                    "erase_node referenced a node that does not exist")
            reason = _protected(node)
            if reason:
                raise CandidateApplicationError(
                    f"refusing to erase {payload['node']}: {reason}")
            graph.erase_node(node)

    validate_graph(working)
    return working


def validate_graph(exported_program) -> None:
    """Structural checks before the graph is allowed anywhere near execution."""
    graph = exported_program.graph

    try:
        graph.lint()
    except Exception as error:
        raise CandidateApplicationError(
            f"the rewritten graph did not pass lint: "
            f"{type(error).__name__}: {str(error)[:200]}")

    outputs = [node for node in graph.nodes if node.op == "output"]
    if len(outputs) != 1:
        raise CandidateApplicationError(
            f"the rewritten graph has {len(outputs)} output nodes")
    if not any(node.op == "placeholder" for node in graph.nodes):
        raise CandidateApplicationError(
            "the rewritten graph has no inputs left")

    try:
        exported_program.graph_module.recompile()
    except Exception as error:
        raise CandidateApplicationError(
            f"the rewritten graph could not be recompiled: "
            f"{type(error).__name__}: {str(error)[:200]}")


def baseline_is_unchanged(baseline_program, node_count: int) -> bool:
    """A cheap assertion that the pristine graph survived an attempt."""
    return len(list(baseline_program.graph.nodes)) == node_count
