"""Mapping a measured runtime operator back to the graph node that produced it.

The Arm target reports what it *executed*: ExecuTorch kernel names, after
lowering, decomposition and out-variant conversion. A repair has to be applied
to what was *exported*: ATen nodes in the `ExportedProgram`. Those two
vocabularies are related but not equal, and the difference is systematic:

    runtime                    exported graph
    native_layer_norm.out  ->  aten::layer_norm  /  aten::native_layer_norm
    expand_copy.out        ->  aten::expand
    where.self_out         ->  aten::where
    _softmax.out           ->  aten::softmax

Substring matching on the runtime name - which is what this replaced - fails on
every row above but the last, because `"expand_copy"` is not a substring of
`"aten::expand"` and `"native_layer_norm"` is not a substring of
`"aten::layer_norm"`. The result was a correctly measured hotspot that repair
could not find, reported as though the operator did not exist.

Two mechanisms, in order of preference
--------------------------------------

1. **Stable identity.** If profiling carried a debug handle through from
   lowering, it names the node directly and no string is involved. ExecuTorch
   only populates `Event.debug_handles` when an ETRecord is supplied to the
   Inspector, so this is usually absent today - but it is read when present,
   and it takes precedence when it is.

2. **Canonical form.** Both names are reduced to a namespace-free, suffix-free,
   variant-free root, and matched on that. One place, one set of rules; nothing
   downstream does string surgery on an operator name.

Conservatism
------------
A runtime operator that matches several graph nodes is *not* resolved by
picking one. Repairing the wrong LayerNorm in a model with twenty-four of them
would be a silent correctness change, so an unresolvable hotspot is reported as
unresolvable and left alone. Ordinal disambiguation is used only when the
number of measured sites equals the number of candidate nodes, which makes the
correspondence a fact about the trace rather than a guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# --- how a runtime name is reduced to a root ---------------------------------

# ExecuTorch's profiling prefix, before anything else is considered.
RUNTIME_EVENT_PREFIX = "native_call_"

# Namespaces that carry no meaning for identity. `torch.ops.` is included
# because `str(OpOverload)` sometimes spells it that way.
_NAMESPACE_PATTERN = re.compile(
    r"^(?:torch\.ops\.)?(?:aten|prims|prim|quantized_decomposed|dim_order_ops)"
    r"(?:::|\.)")

# An out-variant writes into a caller-provided tensor. It is the same operator.
_OUT_SUFFIXES = ("_out", "_outf")

# `_copy` marks a functional variant of an operator that could alias. Same
# operator for our purposes: `expand_copy` repairs at the `expand` node.
_FUNCTIONAL_SUFFIXES = ("_copy",)

# `native_` marks ATen's decomposed implementation of a composite operator.
# `layer_norm` decomposes to `native_layer_norm`; both name the same site.
_IMPLEMENTATION_PREFIXES = ("native_",)


def strip_event_prefix(name: str) -> str:
    """`native_call__softmax.out` -> `_softmax.out`."""
    text = (name or "").strip()
    if text.startswith(RUNTIME_EVENT_PREFIX):
        return text[len(RUNTIME_EVENT_PREFIX):]
    return text


def canonical_operator(name: str) -> str:
    """The comparable root of an operator name, from either vocabulary.

    Applied to both sides of every comparison, so the two never need to agree
    on spelling:

        canonical_operator("native_layer_norm.out")        == "layer_norm"
        canonical_operator("aten::layer_norm")             == "layer_norm"
        canonical_operator("aten.native_layer_norm.default") == "layer_norm"

    Reductions, in order: strip the profiling prefix, strip the namespace, drop
    everything after the first dot (that is the overload - `.default`, `.out`,
    `.self`, `.Tensor`, `.self_out`), then strip out-variant, functional-variant
    and implementation affixes, then leading underscores.

    Deliberately lossy. It is an equivalence-class key, not a name: two
    operators landing in the same class is exactly what makes correlation
    possible, and the ambiguity that creates is handled by the resolver rather
    than by trying to be cleverer here.
    """
    text = strip_event_prefix(name).strip()
    if not text:
        return ""

    text = _NAMESPACE_PATTERN.sub("", text)

    # The overload lives after the first dot and never changes which operator
    # this is: `where.self`, `where.self_out` and `where.default` are one op.
    text = text.split(".", 1)[0]

    changed = True
    while changed:
        changed = False
        for suffix in _OUT_SUFFIXES + _FUNCTIONAL_SUFFIXES:
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[:-len(suffix)]
                changed = True
        for prefix in _IMPLEMENTATION_PREFIXES:
            if text.startswith(prefix) and len(text) > len(prefix):
                text = text[len(prefix):]
                changed = True

    # A leading underscore marks a private ATen spelling of a public operator:
    # `_softmax` is `softmax`.
    return text.lstrip("_").lower()


def operator_target_name(node) -> str:
    """A printable operator name for one graph node, with no object repr."""
    target = getattr(node, "target", None)
    if target is None:
        return ""
    schema = getattr(target, "_schema", None)
    if schema is not None:
        name = str(getattr(schema, "name", ""))
        if name:
            return name
    text = str(target)
    # `<built-in method foo of PyCapsule object at 0x...>` would leak an
    # address and say nothing useful.
    return text if "0x" not in text else getattr(target, "__name__", "operator")


# --- what a resolution attempt produced ---------------------------------------

RESOLVED = "RESOLVED"
AMBIGUOUS = "AMBIGUOUS"
UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class HotspotResolution:
    """Which graph node(s) a measured runtime operator corresponds to.

    A structured answer rather than a string-or-empty, because "no such
    operator" and "seventeen of them" need different reporting and different
    handling, and collapsing them into `""` is what produced a misleading
    "operator was not found in the graph" for an operator that was there
    twenty-four times.
    """

    runtime_operator: str
    canonical: str
    status: str
    node_ids: tuple = ()
    site_costs: tuple = ()
    reason: str = ""
    detail: str = ""

    @property
    def resolved(self) -> bool:
        return self.status == RESOLVED

    @property
    def node_id(self) -> str:
        """The single node this resolved to, or "" when it did not resolve."""
        return self.node_ids[0] if self.resolved and self.node_ids else ""

    @property
    def candidate_count(self) -> int:
        return len(self.node_ids)

    def describe(self) -> str:
        """One line for the terminal. No graph internals."""
        if self.resolved:
            return f"{self.runtime_operator} -> {self.node_id}"
        return f"{self.runtime_operator}: {self.reason}"

    def to_dict(self) -> dict:
        return {
            "runtime_operator": self.runtime_operator,
            "canonical": self.canonical,
            "status": self.status,
            "node_ids": list(self.node_ids),
            "reason": self.reason,
        }


UNRESOLVED_REASON = "could not be correlated to an exported graph node"


def _ambiguous_reason(count: int) -> str:
    return (f"mapped to {count} possible graph nodes and could not be "
            f"resolved uniquely")


# --- resolution ----------------------------------------------------------------


def candidate_nodes(exported_program, runtime_operator: str) -> list:
    """Every graph node whose operator canonicalizes the same way, in order.

    Graph order is execution order, which is what makes ordinal correlation
    against the trace meaningful further down.
    """
    wanted = canonical_operator(runtime_operator)
    if not wanted:
        return []

    try:
        nodes = list(exported_program.graph.nodes)
    except Exception:
        return []

    return [node for node in nodes
            if getattr(node, "op", "") == "call_function"
            and canonical_operator(operator_target_name(node)) == wanted]


def resolve_hotspot(exported_program, runtime_operator: str,
                    debug_node_id: str = "", occurrence: Optional[int] = None,
                    site_count: Optional[int] = None,
                    exclude: frozenset = frozenset()) -> HotspotResolution:
    """Which node does this measured operator refer to?

    `debug_node_id` short-circuits everything: when lowering carried a node
    identity through to the trace, that is the answer and no name is compared.

    `occurrence` and `site_count` come from the profile. When the trace
    measured the same number of sites as the graph has candidate nodes, the
    Nth measured site is the Nth node in execution order - a correspondence
    the trace establishes, not one this function assumes. Any other count is
    left ambiguous.

    `exclude` lets a caller resolve several hotspots against one graph without
    two of them claiming the same node.
    """
    canonical = canonical_operator(runtime_operator)

    if debug_node_id:
        return HotspotResolution(
            runtime_operator=runtime_operator, canonical=canonical,
            status=RESOLVED, node_ids=(debug_node_id,),
            detail="resolved from lowering debug metadata")

    candidates = [str(getattr(node, "name", ""))
                  for node in candidate_nodes(exported_program, runtime_operator)]
    available = [name for name in candidates if name and name not in exclude]

    if not available:
        return HotspotResolution(
            runtime_operator=runtime_operator, canonical=canonical,
            status=UNRESOLVED, reason=UNRESOLVED_REASON,
            detail=(f"no call_function node canonicalizes to {canonical!r}"
                    if not candidates else
                    f"all {len(candidates)} candidate node(s) for "
                    f"{canonical!r} are already claimed"))

    if len(available) == 1:
        return HotspotResolution(
            runtime_operator=runtime_operator, canonical=canonical,
            status=RESOLVED, node_ids=(available[0],),
            detail=f"one node canonicalizes to {canonical!r}")

    # Several candidates. Only a measured one-to-one correspondence resolves
    # this; anything else would be choosing a node to rewrite by coin flip.
    if (occurrence is not None and site_count is not None
            and site_count == len(candidates) and 0 <= occurrence < len(candidates)):
        chosen = candidates[occurrence]
        if chosen not in exclude:
            return HotspotResolution(
                runtime_operator=runtime_operator, canonical=canonical,
                status=RESOLVED, node_ids=(chosen,),
                detail=(f"site {occurrence + 1} of {site_count} matched to "
                        f"graph node {occurrence + 1} of {len(candidates)} in "
                        f"execution order"))

    return HotspotResolution(
        runtime_operator=runtime_operator, canonical=canonical,
        status=AMBIGUOUS, node_ids=tuple(available),
        reason=_ambiguous_reason(len(available)),
        detail=(f"{len(available)} node(s) canonicalize to {canonical!r}; "
                f"the trace measured "
                f"{site_count if site_count is not None else 'an unknown number of'} "
                f"site(s), so execution order does not correspond"))


def resolve_all(exported_program, kernels) -> list:
    """Resolve every profiled kernel against one graph, without collisions.

    Returns `[(kernel, HotspotResolution), ...]` in the order given. A node
    claimed by one kernel is not offered to the next, so two different
    operators that happen to canonicalize alike cannot both target it.
    """
    claimed = set()
    results = []
    for kernel in kernels:
        resolution = resolve_hotspot(
            exported_program,
            getattr(kernel, "operator_name", ""),
            debug_node_id=_debug_node_id_of(kernel),
            site_count=getattr(kernel, "site_count", None),
            exclude=frozenset(claimed),
        )
        if resolution.resolved:
            claimed.add(resolution.node_id)
        results.append((kernel, resolution))
    return results


def _debug_node_id_of(kernel) -> str:
    """A node identity carried through from lowering, when one exists.

    ExecuTorch populates `Event.debug_handles` only when the Inspector is given
    an ETRecord. DelegateDoctor does not generate one today, so this is
    normally empty - it is read anyway, so that wiring an ETRecord in later
    upgrades correlation from canonical-name matching to exact identity with
    no further change here.
    """
    return str(getattr(kernel, "debug_node_id", "") or "")


# --- reporting -----------------------------------------------------------------


def format_resolution_failure(resolution: HotspotResolution,
                              verbose: bool = False) -> str:
    """What to tell the user when a measured hotspot cannot be targeted.

    The headline distinguishes "not found" from "found too many times", because
    they mean different things and lead to different next steps. The graph
    detail is verbose-only: it is diagnostic noise in a normal run.
    """
    text = f"Runtime hotspot {resolution.runtime_operator} {resolution.reason}."
    if verbose and resolution.detail:
        text += f"\n  {resolution.detail}"
        if resolution.node_ids:
            text += f"\n  candidates: {', '.join(resolution.node_ids)}"
    return text


@dataclass
class CorrelationReport:
    """Every resolution in one run, for the verbose log and the artifacts."""

    entries: list = field(default_factory=list)

    def add(self, resolution: HotspotResolution) -> None:
        self.entries.append(resolution)

    @property
    def unresolved(self) -> list:
        return [entry for entry in self.entries if not entry.resolved]

    def to_dict(self) -> dict:
        return {"resolutions": [entry.to_dict() for entry in self.entries]}
