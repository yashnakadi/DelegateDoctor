"""AI repair as a bounded research step over the whole exported model.

The unit of AI repair is the **model**, not a hotspot and not an operator.

Earlier designs asked "find a repair for `native_layer_norm.out`", which forced
every proposal to have the same boundaries as an ETDump event name. Real
opportunities do not respect those boundaries: a Swin block's cost lives across
`expand`, `where`, `reshape`, `permute` and `layer_norm` together, and no
question phrased about one of them can find the transformation that addresses
all five.

So the question asked is:

    Here is the exported graph, here is what it cost on the Arm target, and
    here are the repairs DelegateDoctor already knows. What transformations
    expressible in the constrained DSL would reduce portable execution?

Eligibility is equally simple. Once no known repair applies, the only question
is whether enough of the model still runs outside the delegate to be worth
investigating:

    portable runtime > MIN_AI_PORTABLE_RUNTIME_SHARE

Not "is any single operator above 5%". Three operators at 3%, 2% and 2% are a
7% opportunity, and deciding *which combination* is worth repairing is the
research question - it cannot be a precondition for asking it.

What crosses the boundary
-------------------------
Operator names, stable node identifiers, graph relationships, shapes, dtypes,
DSL-permitted literal arguments, delegated-versus-portable status, measured
runtime, and descriptions of the existing DD rules so a provider does not spend
the request rediscovering DD-001.

Never: model source, weights, checkpoints, tensor values, representative
inputs, filesystem contents, environment variables, credentials. No tools, no
browsing, no code execution - the reply is a structured plan in the existing
constrained DSL and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import operator_correlation

# Is enough of this model still running outside the delegate to be worth a
# bounded investigation? Measured on the CURRENT model, after every known
# repair has already been applied and the model re-profiled.
#
# Deliberately about the model as a whole. A per-operator bar would rule out
# exactly the case AI is useful for: many small fallbacks that share one cause.
MIN_AI_PORTABLE_RUNTIME_SHARE = 0.05

# How much graph is described. Bounded so the request stays a summary rather
# than a serialization of the model: portable regions in full, delegated
# regions as counts, and a hard cap on nodes.
MAX_DESCRIBED_NODES = 120
MAX_LITERALS_PER_NODE = 8
MAX_OPERATOR_SUMMARIES = 12


@dataclass
class ExplorationDecision:
    """Whether the model is worth investigating, and why or why not."""

    portable_runtime_share: float = 0.0
    runtime_delegation_share: float = 0.0
    eligible: bool = False
    reason: str = ""

    def describe(self) -> str:
        return (f"Runtime delegation      "
                f"{100 * self.runtime_delegation_share:.1f}%\n"
                f"Portable runtime        "
                f"{100 * self.portable_runtime_share:.1f}%")


def assess(profile) -> ExplorationDecision:
    """Is there enough portable runtime left to investigate?"""
    if profile is None:
        return ExplorationDecision(reason="no device profile is available")

    delegated = profile.runtime_delegation_fraction
    # Measured directly rather than as `1 - delegated`: the subtraction turns
    # a delegation of exactly 0.95 into a portable share of 0.050000000000000044,
    # which clears a strictly-greater-than 5% bar it should sit exactly on.
    total = profile.total_instruction_ms
    portable = (profile.portable_ms / total) if total else 0.0
    eligible = portable > MIN_AI_PORTABLE_RUNTIME_SHARE

    return ExplorationDecision(
        portable_runtime_share=portable,
        runtime_delegation_share=delegated,
        eligible=eligible,
        reason=("" if eligible else
                f"portable runtime {100 * portable:.1f}% does not exceed "
                f"{100 * MIN_AI_PORTABLE_RUNTIME_SHARE:.0f}%"),
    )


# --- the bounded model description --------------------------------------------


def _literals(node) -> list:
    values = []
    for argument in getattr(node, "args", ()) or ():
        if hasattr(argument, "op"):
            continue
        values.append(_safe_literal(argument))
        if len(values) >= MAX_LITERALS_PER_NODE:
            break
    return values


def _safe_literal(value):
    """Only small plain data. Anything else becomes a type name."""
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, str):
        return value[:32]
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_LITERALS_PER_NODE:
            return f"<len {len(value)}>"
        return [_safe_literal(item) for item in value]
    return f"<{type(value).__name__}>"


def _shape_and_dtype(node) -> tuple:
    meta = getattr(node, "meta", None) or {}
    value = meta.get("val")
    shape = getattr(value, "shape", None)
    if shape is None:
        return [], ""
    try:
        dimensions = [int(size) if isinstance(size, int) else "dyn"
                      for size in shape]
    except TypeError:
        return [], ""
    return dimensions, str(getattr(value, "dtype", "") or "")


def describe_graph(exported_program, portable_operators: set = frozenset(),
                   limit: int = MAX_DESCRIBED_NODES) -> list:
    """Every computation node, described in terms safe to transmit.

    Node identifiers are the same `node_N` positions the repair DSL uses, so a
    proposal can reference anything it was shown and nothing it was not.
    """
    try:
        nodes = list(exported_program.graph.nodes)
    except Exception:
        return []

    identifiers = {node: f"node_{index}" for index, node in enumerate(nodes)}
    described = []

    for node in nodes:
        kind = str(getattr(node, "op", ""))
        shape, dtype = _shape_and_dtype(node)

        if kind != "call_function":
            # Placeholders and outputs are described by their *role*, never by
            # their target - a placeholder's target is the parameter's name,
            # and `p_blocks_3_norm_weight` is model structure the request has
            # no reason to carry. They are still listed, because a rewrite may
            # legitimately need to insert before the output or read an input.
            entry = {"id": identifiers[node], "role": kind}
            if shape:
                entry["shape"] = shape
            if dtype:
                entry["dtype"] = dtype
            described.append(entry)
            if len(described) >= limit:
                break
            continue

        target = operator_correlation.operator_target_name(node)
        entry = {
            "id": identifiers[node],
            "target": target,
            "inputs": [identifiers[argument]
                       for argument in getattr(node, "args", ()) or ()
                       if argument in identifiers],
            "users": [identifiers[user] for user in getattr(node, "users", ())
                      if user in identifiers],
        }
        literals = _literals(node)
        if literals:
            entry["literals"] = literals
        if shape:
            entry["shape"] = shape
        if dtype:
            entry["dtype"] = dtype
        if operator_correlation.canonical_operator(target) in portable_operators:
            # Which regions XNNPACK declined is the single most useful thing in
            # here: it is where the remaining time is.
            entry["portable"] = True
        described.append(entry)
        if len(described) >= limit:
            break

    return described


def describe_runtime(profile, delegation=None) -> dict:
    """What the Arm target measured. Numbers only, no tensors."""
    if profile is None:
        return {}

    kernels = sorted(profile.portable_kernels,
                     key=lambda kernel: kernel.total_ms, reverse=True)
    summary = {
        "runtime_delegation": round(profile.runtime_delegation_fraction, 4),
        "portable_runtime": round(
            (profile.portable_ms / profile.total_instruction_ms)
            if profile.total_instruction_ms else 0.0, 4),
        "method_execute_ms": round(profile.method_execute_ms, 3),
        "portable_ms": round(profile.portable_ms, 3),
        "portable_operators": [
            {
                "operator": kernel.operator_name,
                "total_ms": round(kernel.total_ms, 3),
                "runtime_fraction": round(kernel.runtime_fraction, 4),
                "sites": getattr(kernel, "site_count", kernel.call_count),
            }
            for kernel in kernels[:MAX_OPERATOR_SUMMARIES]
        ],
    }
    if delegation is not None:
        summary["operator_delegation"] = round(
            delegation.operator_delegation_fraction, 4)
        summary["delegate_blobs"] = delegation.delegate_blob_count
        summary["total_operators"] = delegation.total_ops
        summary["portable_operator_count"] = delegation.portable_op_total
    return summary


def describe_known_repairs(rules) -> list:
    """What DelegateDoctor already knows, so a request does not rediscover it.

    Sent deliberately: a provider that proposes DD-001 has spent the user's
    money reinventing a rule that already ran and was already applied.
    """
    described = []
    for rule in rules:
        try:
            described.append({
                "id": rule.RULE_ID,
                "title": rule.RULE_TITLE,
                "rewrite": rule.describe_rewrite(),
                "status": "already applied where it matched",
            })
        except Exception:
            continue
    return described


def build_model_context(exported_program, profile, delegation, rules,
                        executorch_version: str = "") -> tuple:
    """(context, known_node_ids) for one model-level exploration.

    Assembled field by field rather than by serializing an object, so adding a
    field to a result class can never silently start transmitting it.
    """
    portable_operators = set()
    if profile is not None:
        portable_operators = {
            operator_correlation.canonical_operator(kernel.operator_name)
            for kernel in profile.portable_kernels}

    graph = describe_graph(exported_program, portable_operators)
    context = {
        "backend": "ExecuTorch XNNPACK",
        "executorch_version": executorch_version,
        "task": ("Analyze this exported model and its measured Arm64 "
                 "execution. Identify graph transformations expressible in "
                 "DelegateDoctor's constrained repair DSL that may reduce "
                 "portable (non-delegated) execution while preserving model "
                 "semantics. A transformation may involve several operators; "
                 "it does not have to correspond to one profiled operator."),
        "graph": {
            "nodes": graph,
            "described_nodes": len(graph),
            "total_graph_nodes": _node_count(exported_program),
        },
        "measurement": describe_runtime(profile, delegation),
        "known_repairs": describe_known_repairs(rules),
    }
    known = [entry["id"] for entry in graph]
    return context, known


def _node_count(exported_program) -> int:
    try:
        return len(list(exported_program.graph.nodes))
    except Exception:
        return 0


# --- terminal ------------------------------------------------------------------


def format_enabled_screen(decision: ExplorationDecision,
                          provider_label: str = "") -> str:
    """What is about to happen, once the user has opted in on the command line.

    Not a prompt. `--ai-repair` was the decision; this states the measurement
    that makes the investigation worth starting, and names the provider.
    """
    lines = ["", decision.describe(), "",
             "No known DelegateDoctor repairs remain.", "",
             "Experimental AI repair enabled (--ai-repair)."]
    if provider_label:
        lines.append(f"Provider                {provider_label}")
    return "\n".join(lines)


def format_exploration_start() -> str:
    return "\nExploring model for experimental AI repairs..."


def format_skipped(decision: ExplorationDecision) -> str:
    """Why the model was not investigated. Only shown when it is close."""
    return (f"\nAI exploration          skipped\n"
            f"Reason                  {decision.reason}")
