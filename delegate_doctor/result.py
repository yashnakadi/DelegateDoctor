"""What DelegateDoctor found, and how far it got.

DelegateDoctor is an analyzer first and an optimizer second. A model does not
have to be repairable - or even runnable on the Arm target - for the run to have
produced useful work. So the pipeline is a sequence of stages, each of which
reports its own outcome, and the run ends with whichever result the stages
actually earned.

The distinction this module exists to keep is:

    "this model failed"              vs   "this stage could not run"
    "DelegateDoctor cannot export"   vs   "ExecuTorch cannot lower"
    "the graph is unanalyzable"      vs   "the device cannot take these inputs"

Collapsing those into "unsupported model" is what this replaces.
"""

from __future__ import annotations

import os
import pathlib
import webbrowser
from dataclasses import dataclass, field
from typing import Optional

# --- stage outcomes ---------------------------------------------------------

PASS = "PASS"
FAILED = "FAILED"          # the stage ran and the stack rejected the graph
UNSUPPORTED = "UNSUPPORTED"  # DelegateDoctor cannot do this yet, for a stated reason
UNAVAILABLE = "UNAVAILABLE"  # the environment cannot do it (no device, no runner)
NOT_RUN = "NOT RUN"        # an earlier stage stopped the pipeline
NONE_FOUND = "NONE"        # the stage ran and found nothing, which is not a failure

# --- stage names ------------------------------------------------------------

EXPORT = "PyTorch export"
GRAPH = "Graph inspection"
LOWERING = "ExecuTorch lowering"
DELEGATION = "XNNPACK analysis"
DEVICE = "Android execution"
PROFILING = "Runtime profiling"
REPAIR = "Repair matching"
VERIFICATION = "Correctness verification"
BENCHMARK = "Device benchmark"

STAGE_ORDER = (EXPORT, GRAPH, LOWERING, DELEGATION, DEVICE, PROFILING,
               REPAIR, VERIFICATION, BENCHMARK)

# --- top-level outcomes -----------------------------------------------------

ANALYSIS_COMPLETE = "ANALYSIS_COMPLETE"
FULLY_DELEGATED = "FULLY_DELEGATED"
NO_REPAIR_REQUIRED = "NO_REPAIR_REQUIRED"
NO_REPAIR_AVAILABLE = "NO_REPAIR_AVAILABLE"
REPAIR_ACCEPTED = "REPAIR_ACCEPTED"
REPAIR_REJECTED = "REPAIR_REJECTED"
EXECUTORCH_LOWERING_UNSUPPORTED = "EXECUTORCH_LOWERING_UNSUPPORTED"
DEVICE_EXECUTION_UNSUPPORTED = "DEVICE_EXECUTION_UNSUPPORTED"

# Human-readable one-liners for the final RESULT block.
OUTCOME_TEXT = {
    ANALYSIS_COMPLETE: "Analysis complete.",
    FULLY_DELEGATED: "Fully delegated to XNNPACK. No repair required.",
    NO_REPAIR_REQUIRED: "No portable hotspot found. No repair required.",
    NO_REPAIR_AVAILABLE: "No repair in the catalog matches this model.",
    REPAIR_ACCEPTED: "Repair verified and faster on the target. Accepted.",
    REPAIR_REJECTED: "Repair rejected: it did not pass every gate.",
    EXECUTORCH_LOWERING_UNSUPPORTED: (
        "The graph exported, but ExecuTorch could not lower it."
    ),
    DEVICE_EXECUTION_UNSUPPORTED: (
        "Static analysis complete. The Arm target could not run this model."
    ),
}

# A rejected repair is a real finding, not a tool error, so only the repair
# decision distinguishes 0 from 1. Genuine failures exit 2 from the CLI.
_NONZERO_EXIT = {REPAIR_REJECTED: 1}


def exit_code(status: str) -> int:
    """CLI exit status. Every successful analysis is 0, whatever it concluded."""
    return _NONZERO_EXIT.get(status, 0)


@dataclass
class Stage:
    """One step of the pipeline and what became of it."""

    name: str
    status: str
    detail: str = ""

    @property
    def ran(self) -> bool:
        return self.status in (PASS, FAILED, NONE_FOUND)

    def to_dict(self) -> dict:
        return {"stage": self.name, "status": self.status, "detail": self.detail}


@dataclass
class OptimizationResult:
    """Everything one DelegateDoctor run learned about one model.

    Returned by the Python API and by the CLI's pipeline. Fields that a stage
    never reached stay None - deliberately, so nothing can read a number that
    was not measured.
    """

    status: str
    model_name: str = ""
    description: str = ""
    stages: list = field(default_factory=list)

    # static analysis
    before_delegation: object = None
    after_delegation: object = None

    # measured on the device, or None when profiling did not run
    before_profile: object = None
    after_profile: object = None

    # repair
    detections: dict = field(default_factory=dict)   # rule id -> DetectionResult
    repairs_applied: dict = field(default_factory=dict)  # rule id -> site count

    # gates
    host_verification: object = None
    device_verification: object = None
    benchmark: object = None
    decision: object = None

    # the Arm target that produced any measurement here, if one did
    device_description: str = ""
    device_is_emulator: bool = False

    # Rule id -> {"title", "rewrite"}, copied from the catalog so the report can
    # describe a repair without importing the rules or knowing what they are.
    repair_catalog: dict = field(default_factory=dict)

    # artifacts
    run_dir: str = ""
    output_pte: Optional[str] = None
    report_text: str = ""
    report_path: Optional[str] = None

    # --- the questions callers actually ask --------------------------------

    @property
    def repair_available(self) -> bool:
        """Did any catalog rule recognise a pattern in this graph?"""
        return any(found.applies for found in self.detections.values())

    @property
    def repair_accepted(self) -> bool:
        return self.status == REPAIR_ACCEPTED

    @property
    def analyzed(self) -> bool:
        """Did the run produce a usable analysis?

        True for every outcome here: even a lowering failure tells you something
        specific about deploying this model with ExecuTorch.
        """
        return bool(self.stages) and self.stage_status(EXPORT) == PASS

    @property
    def summary(self) -> str:
        return OUTCOME_TEXT.get(self.status, self.status)

    @property
    def exit_code(self) -> int:
        return exit_code(self.status)

    def open_report(self) -> bool:
        """Open the HTML report in the developer's normal browser.

        Never raises and never affects the analysis: a headless machine, a
        missing report or a browser that refuses to launch all just return
        False after printing the path, which is the useful thing anyway.
        """
        if not self.report_path or not os.path.isfile(self.report_path):
            print("No HTML report was written for this run.")
            return False

        url = pathlib.Path(self.report_path).absolute().as_uri()
        try:
            opened = webbrowser.open(url)
        except Exception:
            opened = False
        if not opened:
            print(f"Could not open a browser. The report is at:\n  {self.report_path}")
        return bool(opened)

    def stage(self, name: str) -> Optional[Stage]:
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None

    def stage_status(self, name: str) -> str:
        found = self.stage(name)
        return found.status if found else NOT_RUN

    def record(self, name: str, status: str, detail: str = "") -> Stage:
        """Add or replace a stage outcome, keeping STAGE_ORDER."""
        stage = Stage(name=name, status=status, detail=detail)
        for index, existing in enumerate(self.stages):
            if existing.name == name:
                self.stages[index] = stage
                return stage
        self.stages.append(stage)
        self.stages.sort(key=lambda s: STAGE_ORDER.index(s.name)
                         if s.name in STAGE_ORDER else len(STAGE_ORDER))
        return stage

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "summary": self.summary,
            "model": self.model_name,
            "description": self.description,
            "stages": [stage.to_dict() for stage in self.stages],
            "repair_available": self.repair_available,
            "repairs_applied": dict(self.repairs_applied),
            "device": self.device_description,
            "device_is_emulator": self.device_is_emulator,
            "run_dir": self.run_dir,
            "output_pte": self.output_pte,
            "report_path": self.report_path,
        }

    def __repr__(self) -> str:
        return (f"OptimizationResult(status={self.status!r}, "
                f"model={self.model_name!r}, "
                f"repair_available={self.repair_available})")
