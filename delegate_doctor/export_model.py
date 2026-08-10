"""Export a PyTorch model into an ExecuTorch program lowered with XNNPACK.

This is the entry point of the DelegateDoctor pipeline. Everything downstream
(delegation analysis, profiling, repair, verification, benchmarking) works on
the objects produced here.

A note on where in the pipeline things happen, because it matters:

    torch.nn.Module
        -> torch.export.export(...)      produces an ExportedProgram (ATen ops)
        -> to_edge_transform_and_lower() runs the XNNPACK partitioner
        -> to_executorch()               produces the .pte bytes

DD-001 rewrites the graph at the **ExportedProgram** stage, before lowering.
That is the last point where the graph is still ordinary ATen operators that we
can safely edit. Once a .pte exists the delegated regions are opaque compiled
blobs, so DelegateDoctor cannot repair a .pte file directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower


@dataclass
class ModelSpec:
    """Everything DelegateDoctor needs to know about a model to work on it.

    Example modules build and return one of these. Keeping it in a small
    dataclass means an example file never has to import the rest of the tool.
    """

    name: str
    model: torch.nn.Module
    example_inputs: Tuple[torch.Tensor, ...]

    # Dimension to take an argmax over during numerical verification. For a
    # segmentation model this is the class dimension, so "argmax agreement"
    # means "every pixel still gets the same predicted class". Set to None for
    # models where an argmax is not meaningful.
    argmax_dim: Optional[int] = None

    # Free-text description shown in the report header.
    description: str = ""


@dataclass
class ExportResult:
    """The three artifacts of one export, kept together so callers stay simple."""

    exported_program: torch.export.ExportedProgram  # ATen graph, before lowering
    edge_program_manager: object                    # after XNNPACK partitioning
    pte_path: str                                   # serialized ExecuTorch program
    extras: dict = field(default_factory=dict)


def export_to_aten(model: torch.nn.Module, example_inputs) -> torch.export.ExportedProgram:
    """Trace the model into an ExportedProgram of ATen operators.

    This is the graph DD-001 inspects and rewrites.
    """
    model = model.eval()
    return torch.export.export(model, example_inputs, strict=True)


def lower_with_xnnpack(exported_program: torch.export.ExportedProgram):
    """Run the XNNPACK partitioner over an ExportedProgram.

    The partitioner decides, operator by operator, what it is willing to take.
    Anything it refuses stays in the graph and will run on ExecuTorch's portable
    (reference C++) kernels at inference time. Those refusals are exactly what
    DelegateDoctor hunts for.
    """
    return to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
        # The IR validity checker is strict about some legal-but-unusual graphs.
        # Turning it off matches what the ExecuTorch examples do.
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )


def write_pte(edge_program_manager, pte_path: str) -> str:
    """Serialize a lowered program to a .pte file on disk."""
    parent = os.path.dirname(pte_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    executorch_program = edge_program_manager.to_executorch()
    with open(pte_path, "wb") as pte_file:
        executorch_program.write_to_file(pte_file)
    return pte_path


def export_and_lower(
    model: torch.nn.Module,
    example_inputs,
    pte_path: str,
) -> ExportResult:
    """Run the whole export pipeline for a model that needs no repair."""
    exported_program = export_to_aten(model, example_inputs)
    return lower_and_write(exported_program, pte_path)


def lower_and_write(
    exported_program: torch.export.ExportedProgram,
    pte_path: str,
) -> ExportResult:
    """Lower an (already traced, possibly repaired) ExportedProgram and save it.

    DD-001 hands us a modified ExportedProgram, so the repaired path re-enters
    the pipeline here rather than re-tracing the model.
    """
    edge_program_manager = lower_with_xnnpack(exported_program)

    # to_executorch() mutates the manager, so the readable graph dumps and the
    # delegation analysis must happen against a copy taken beforehand. We keep
    # the pre-mutation manager and write the .pte from a deep copy.
    import copy

    manager_for_analysis = copy.deepcopy(edge_program_manager)
    write_pte(edge_program_manager, pte_path)

    return ExportResult(
        exported_program=exported_program,
        edge_program_manager=manager_for_analysis,
        pte_path=pte_path,
    )


def save_readable_graphs(export_result: ExportResult, output_dir: str) -> None:
    """Write human-readable text dumps of the graphs, for the run artifacts."""
    os.makedirs(output_dir, exist_ok=True)

    aten_path = os.path.join(output_dir, "exported_graph.txt")
    with open(aten_path, "w") as aten_file:
        aten_file.write(export_result.exported_program.graph_module.print_readable(
            print_output=False
        ))

    lowered_path = os.path.join(output_dir, "lowered_graph.txt")
    graph_module = export_result.edge_program_manager.exported_program().graph_module
    with open(lowered_path, "w") as lowered_file:
        lowered_file.write(graph_module.print_readable(print_output=False))


def run_on_host(pte_path: str, inputs) -> list:
    """Execute a .pte through ExecuTorch's Python runtime bindings.

    Used for numerical verification, where we want a fast, deterministic run and
    do not care about latency. Benchmarking uses the Arm device instead.
    """
    from executorch.runtime import Runtime

    program = Runtime.get().load_program(pte_path)
    method = program.load_method("forward")
    return method.execute(list(inputs))
