"""Operator-count delegation: how much of the graph did XNNPACK take?

This is the number most tools report, and the feasibility study showed it can be
badly misleading on its own (a model at 96.8% operator delegation spent 65% of
its time in the 3.2% that was left over). We compute it because it is useful
context, but `profiling.py` computes the number that actually matters.

Nothing here is guessed from operator names. We walk the lowered program:

  * a node calling `executorch_call_delegate` is one XNNPACK blob;
  * the operators *inside* that blob live on the attached LoweredBackendModule;
  * every other `call_function` node in the top-level graph is a portable
    operator that will run on ExecuTorch's reference C++ kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

DELEGATE_CALL_TARGET = "executorch_call_delegate"

# Nodes that do not perform model arithmetic. `alloc` comes from memory
# planning and `getitem` just unpacks tuples; counting them would distort the
# percentages without telling us anything about the model.
NON_COMPUTE_OPS = {"alloc", "getitem", DELEGATE_CALL_TARGET}


@dataclass
class DelegationReport:
    """Operator counts for one lowered program."""

    delegate_blob_count: int
    delegated_op_counts: Dict[str, int] = field(default_factory=dict)
    portable_op_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def delegated_op_total(self) -> int:
        return sum(self.delegated_op_counts.values())

    @property
    def portable_op_total(self) -> int:
        return sum(self.portable_op_counts.values())

    @property
    def total_ops(self) -> int:
        return self.delegated_op_total + self.portable_op_total

    @property
    def operator_delegation_fraction(self) -> float:
        """Fraction of operators XNNPACK accepted, between 0.0 and 1.0."""
        if self.total_ops == 0:
            return 0.0
        return self.delegated_op_total / self.total_ops


def _operator_name(node: torch.fx.Node) -> str:
    """A readable name for an fx node's target, e.g. 'aten::_softmax'."""
    target = node.target
    name = getattr(target, "_name", None)
    if name is None:
        name = getattr(target, "__name__", None)
    if name is None:
        name = str(target)
    return name


def _is_delegate_call(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and DELEGATE_CALL_TARGET in str(node.target)


def _is_compute_op(name: str) -> bool:
    # Names arrive as either 'alloc' or 'aten::add.out'; check the bare head.
    return name not in NON_COMPUTE_OPS and name.split(".")[0] not in NON_COMPUTE_OPS


def _find_lowered_modules(graph_module: torch.fx.GraphModule) -> List[object]:
    """Return the LoweredBackendModule objects hanging off a graph module.

    Each one holds the sub-graph that was handed to XNNPACK. We match on the
    class name rather than importing the class, because its import path has
    moved between ExecuTorch versions.
    """
    lowered_modules = []
    for _, submodule in graph_module.named_modules():
        if type(submodule).__name__ == "LoweredBackendModule":
            lowered_modules.append(submodule)
    return lowered_modules


def analyze_delegation(edge_program_manager) -> DelegationReport:
    """Count delegated vs portable operators in a lowered program."""
    graph_module = edge_program_manager.exported_program().graph_module

    delegate_blob_count = 0
    portable_op_counts: Dict[str, int] = {}
    delegated_op_counts: Dict[str, int] = {}

    # Top-level graph: delegate calls plus whatever XNNPACK refused.
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        if _is_delegate_call(node):
            delegate_blob_count += 1
            continue
        name = _operator_name(node)
        if _is_compute_op(name):
            portable_op_counts[name] = portable_op_counts.get(name, 0) + 1

    # Inside each delegate blob: the operators XNNPACK accepted.
    for lowered_module in _find_lowered_modules(graph_module):
        blob_graph = lowered_module.original_module.graph_module
        for node in blob_graph.graph.nodes:
            if node.op != "call_function":
                continue
            name = _operator_name(node)
            if _is_compute_op(name):
                delegated_op_counts[name] = delegated_op_counts.get(name, 0) + 1

    return DelegationReport(
        delegate_blob_count=delegate_blob_count,
        delegated_op_counts=delegated_op_counts,
        portable_op_counts=portable_op_counts,
    )
