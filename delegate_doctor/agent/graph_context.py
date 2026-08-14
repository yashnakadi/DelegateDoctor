"""A bounded, sanitized view of the graph around a measured hotspot.

What an AI needs to reason about a backend rejection is structure: which
operator, on what shapes, fed by what. It does not need the weights, the tensor
values, the representative inputs, or the user's Python.

So this module builds a small neighbourhood - N nodes either side of the
hotspot - described only in terms of:

    a stable local identifier (node_0, node_1, ...)
    the operator target
    which other nodes feed it
    literal arguments
    tensor rank, shape and dtype from metadata

Nothing else crosses. Constants are reported as *shapes*, never values, so a
parameter tensor cannot leave the machine one literal at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import operator_correlation

# How far either side of the hotspot to look. Small on purpose: the question is
# local, and a bigger window is more of the user's model leaving the machine.
NEIGHBOURHOOD_RADIUS = 6
MAX_NEIGHBOURHOOD_NODES = 2 * NEIGHBOURHOOD_RADIUS + 1

# Literal arguments above this size are summarized rather than sent verbatim.
MAX_LITERAL_ITEMS = 16


@dataclass
class NodeView:
    """One graph node, described in terms safe to transmit."""

    identifier: str
    op: str
    target: str
    inputs: list = field(default_factory=list)
    literals: list = field(default_factory=list)
    shape: list = field(default_factory=list)
    dtype: str = ""
    is_hotspot: bool = False

    def to_dict(self) -> dict:
        payload = {"id": self.identifier, "op": self.op, "target": self.target}
        if self.inputs:
            payload["inputs"] = self.inputs
        if self.literals:
            payload["literals"] = self.literals
        if self.shape:
            payload["shape"] = self.shape
        if self.dtype:
            payload["dtype"] = self.dtype
        if self.is_hotspot:
            payload["hotspot"] = True
        return payload


@dataclass
class GraphNeighbourhood:
    """The window around the hotspot, plus the measured facts about it."""

    nodes: list = field(default_factory=list)
    hotspot_identifier: str = ""
    hotspot_operator: str = ""
    total_graph_nodes: int = 0

    def to_dict(self) -> dict:
        return {
            "hotspot": self.hotspot_identifier,
            "hotspot_operator": self.hotspot_operator,
            "total_graph_nodes": self.total_graph_nodes,
            "nodes": [node.to_dict() for node in self.nodes],
        }


def _target_name(node) -> str:
    """A printable operator name, with no object repr and no memory address.

    Shared with `operator_correlation`, so the name a node is *described* by
    and the name it is *matched* by can never drift apart.
    """
    if getattr(node, "target", None) is None:
        return str(getattr(node, "op", ""))
    return operator_correlation.operator_target_name(node)


def _safe_literal(value, depth: int = 0):
    """Only small plain data. Anything else becomes a type name."""
    if depth > 2:
        return "..."
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str) and len(value) > 64:
            return f"<str len={len(value)}>"
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else "<nonfinite>"
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_LITERAL_ITEMS:
            return f"<sequence len={len(value)}>"
        return [_safe_literal(item, depth + 1) for item in value]
    return f"<{type(value).__name__}>"


def _tensor_metadata(node) -> tuple:
    """(shape, dtype) from node metadata. Never the tensor's values."""
    meta = getattr(node, "meta", {}) or {}
    value = meta.get("val")
    shape, dtype = [], ""
    try:
        if value is not None and hasattr(value, "shape"):
            shape = [int(dimension) if isinstance(dimension, int) else -1
                     for dimension in value.shape]
            dtype = str(getattr(value, "dtype", "")).replace("torch.", "")
    except Exception:
        return [], ""
    return shape, dtype


def build_neighbourhood(exported_program, hotspot_operator: str,
                        radius: int = NEIGHBOURHOOD_RADIUS,
                        node_name: str = "") -> GraphNeighbourhood:
    """Find the hotspot node and describe a bounded window around it.

    `hotspot_operator` is the portable kernel name from profiling, e.g.
    `_softmax.out`; matching is on the operator's short name so ETDump's
    spelling and the graph's need not be identical.

    `node_name` pins the choice to one exact graph node. The same operator can
    appear several times in a graph, and when the repair loop is working
    through hotspots one at a time, "the first `mean`" is not good enough - it
    would describe the wrong neighbourhood for every occurrence but the first.
    """
    nodes = list(exported_program.graph.nodes)
    identifiers = {node: f"node_{index}" for index, node in enumerate(nodes)}
    hotspot_index = None

    # A resolved node name is the answer, and the only one that distinguishes
    # the third LayerNorm from the first.
    if node_name:
        for index, node in enumerate(nodes):
            if str(getattr(node, "name", "")) == node_name:
                hotspot_index = index
                break

    # Without one, fall back to canonical operator matching - the same rules
    # the resolver uses, so this cannot disagree with it. Only an unambiguous
    # single match is accepted: guessing which site to describe would send the
    # wrong neighbourhood to the provider.
    if hotspot_index is None:
        wanted = operator_correlation.canonical_operator(hotspot_operator)
        matches = [index for index, node in enumerate(nodes)
                   if getattr(node, "op", "") == "call_function"
                   and wanted
                   and operator_correlation.canonical_operator(
                       _target_name(node)) == wanted]
        if len(matches) == 1:
            hotspot_index = matches[0]

    if hotspot_index is None:
        return GraphNeighbourhood(total_graph_nodes=len(nodes))

    start = max(0, hotspot_index - radius)
    end = min(len(nodes), hotspot_index + radius + 1)

    views = []
    for index in range(start, end):
        node = nodes[index]
        shape, dtype = _tensor_metadata(node)
        inputs, literals = [], []
        for argument in getattr(node, "args", ()):  # positional only
            if argument in identifiers:
                inputs.append(identifiers[argument])
            else:
                literals.append(_safe_literal(argument))
        views.append(NodeView(
            identifier=identifiers[node],
            op=str(getattr(node, "op", "")),
            target=_target_name(node),
            inputs=inputs,
            literals=literals,
            shape=shape,
            dtype=dtype,
            is_hotspot=(index == hotspot_index),
        ))

    return GraphNeighbourhood(
        nodes=views,
        hotspot_identifier=identifiers[nodes[hotspot_index]],
        hotspot_operator=_target_name(nodes[hotspot_index]),
        total_graph_nodes=len(nodes),
    )


def build_repair_context(neighbourhood: GraphNeighbourhood, profile,
                         delegation, executorch_version: str = "",
                         hotspot=None) -> dict:
    """Everything a repair request may contain, and nothing else.

    Assembled explicitly rather than by serializing an object, so adding a
    field to a result class can never silently start transmitting it.

    `hotspot` names which portable kernel this request is about. It defaults to
    the most expensive one, which is right for a single-shot request and wrong
    for the second iteration of the repair loop - by then the caller is asking
    about a different operator and the measurement must say so.
    """
    if hotspot is None:
        hotspot = profile.portable_kernels[0] if profile.portable_kernels else None
    return {
        "backend": "ExecuTorch XNNPACK",
        "executorch_version": executorch_version,
        "graph": neighbourhood.to_dict(),
        "measurement": {
            "hotspot_operator": hotspot.operator_name if hotspot else "",
            "hotspot_ms": round(hotspot.total_ms, 3) if hotspot else 0.0,
            "hotspot_runtime_fraction": round(hotspot.runtime_fraction, 4)
            if hotspot else 0.0,
            "runtime_delegation": round(profile.runtime_delegation_fraction, 4),
            "operator_delegation": round(
                delegation.operator_delegation_fraction, 4),
            "portable_operators": delegation.portable_op_total,
            "total_operators": delegation.total_ops,
            "delegate_blobs": delegation.delegate_blob_count,
        },
    }
