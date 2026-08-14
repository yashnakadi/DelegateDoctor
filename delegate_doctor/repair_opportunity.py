"""How much is actually on the table, worked out once.

Before DelegateDoctor asks whether to spend an API request exploring a repair,
the user deserves enough context to answer. "3.6% of measured runtime" is not
enough: it does not say whether fallback matters at all, whether this hotspot
dominates it, or what the best imaginable outcome would be.

Every number below is derived here and nowhere else, so the terminal prompt,
`report.txt` and `report.html` cannot drift apart or disagree.

Two metric families are kept apart on purpose:

  * **Benchmark p50 latency** - wall time for the whole model, measured by the
    tracer-free runner.
  * **ETDump event time** - per-operator durations from the profiling runner.

They are different measurements of different things and will not add up to each
other. Portable milliseconds come from the profiler's own accumulated portable
event time, never from multiplying a p50 by a percentage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Below this share of measured runtime, removing the hotspot entirely could not
# change much, and the user should be told so before paying for a request.
SMALL_HOTSPOT_FRACTION = 0.05

# How many secondary hotspots to name. Enough to see whether the problem is one
# operator or spread out; not the whole profiler table.
OTHER_HOTSPOT_LIMIT = 3

AI_NOT_REQUESTED = "Not requested"
AI_DECLINED = "Declined"
AI_UNAVAILABLE = "Unavailable"
AI_ATTEMPTED = "Attempted"


@dataclass(frozen=True)
class HotspotView:
    """One portable operator, with its measured cost expressed three ways."""

    operator: str
    runtime_ms: float
    total_fraction: float          # share of all measured runtime
    portable_fraction: float       # share of the non-delegated runtime

    def describe(self) -> str:
        return (f"{self.operator}  {self.runtime_ms:.3f} ms · "
                f"{100 * self.total_fraction:.1f}%")


@dataclass
class RepairOpportunitySummary:
    """What a repair could be worth, and where the cost actually is."""

    target: str = ""
    measured_latency_ms: Optional[float] = None

    operator_delegation: Optional[float] = None     # 0.0 - 1.0
    runtime_delegation: Optional[float] = None      # 0.0 - 1.0

    portable_runtime_ms: Optional[float] = None
    portable_operator_count: int = 0

    top_hotspot: Optional[HotspotView] = None
    other_hotspots: list = field(default_factory=list)

    catalog_match: str = "None"
    ai_status: str = AI_NOT_REQUESTED
    provider_label: str = ""
    provider_model: str = ""
    provider_is_local: bool = False

    # --- derived ------------------------------------------------------------

    @property
    def portable_runtime_fraction(self) -> Optional[float]:
        """The share of measured runtime spent outside the delegate."""
        if self.runtime_delegation is None:
            return None
        return max(0.0, 1.0 - self.runtime_delegation)

    @property
    def theoretical_upper_bound_speedup(self) -> Optional[float]:
        """Amdahl's ceiling if the top hotspot's cost became exactly zero.

        An upper bound, not a prediction and not an expectation: it assumes the
        operator becomes free and that nothing else changes, neither of which a
        real rewrite achieves. Returned as None when the share is unknown or
        pathological, because a fabricated number here would be worse than a
        missing one.
        """
        if self.top_hotspot is None:
            return None
        fraction = self.top_hotspot.total_fraction
        if fraction is None or fraction <= 0.0 or fraction >= 1.0:
            return None
        return 1.0 / (1.0 - fraction)

    @property
    def hotspot_is_small(self) -> bool:
        return (self.top_hotspot is not None
                and 0.0 < self.top_hotspot.total_fraction < SMALL_HOTSPOT_FRACTION)

    @property
    def has_measurement(self) -> bool:
        return self.runtime_delegation is not None

    def to_dict(self) -> dict:
        """For results.json. Percentages as fractions, milliseconds as measured."""
        return {
            "target": self.target,
            "measured_latency_ms": self.measured_latency_ms,
            "operator_delegation": self.operator_delegation,
            "runtime_delegation": self.runtime_delegation,
            "portable_runtime_fraction": self.portable_runtime_fraction,
            "portable_runtime_ms": self.portable_runtime_ms,
            "portable_operator_count": self.portable_operator_count,
            "top_hotspot": ({
                "operator": self.top_hotspot.operator,
                "runtime_ms": self.top_hotspot.runtime_ms,
                "total_fraction": self.top_hotspot.total_fraction,
                "portable_fraction": self.top_hotspot.portable_fraction,
            } if self.top_hotspot else None),
            "other_hotspots": [
                {"operator": hotspot.operator, "runtime_ms": hotspot.runtime_ms,
                 "total_fraction": hotspot.total_fraction}
                for hotspot in self.other_hotspots
            ],
            "theoretical_upper_bound_speedup":
                self.theoretical_upper_bound_speedup,
            "catalog_match": self.catalog_match,
            "ai_status": self.ai_status,
            "ai_provider": self.provider_label or None,
            "ai_model": self.provider_model or None,
        }


def build_summary(profile=None, delegation=None, target: str = "",
                  catalog_match: str = "None", configuration=None,
                  ai_status: str = AI_NOT_REQUESTED,
                  focus_kernel: str = "") -> RepairOpportunitySummary:
    """Assemble the summary from what was actually measured.

    Nothing is invented: a field whose source is missing stays None, and the
    renderers omit it rather than printing a zero that looks measured.

    `focus_kernel` names which portable kernel the screen is about. It defaults
    to the most expensive one, which is right for the run-level summary and
    wrong for the repair loop's third iteration - by then the question is about
    a specific hotspot, and the ceiling shown must be that hotspot's.
    """
    summary = RepairOpportunitySummary(
        target=target, catalog_match=catalog_match, ai_status=ai_status)

    if delegation is not None:
        summary.operator_delegation = delegation.operator_delegation_fraction

    if profile is not None:
        # Method::execute from the same ETDump the hotspots came from, so every
        # millisecond on this screen belongs to one measurement family. The
        # benchmark's p50 is a different runner and is deliberately not mixed in.
        summary.measured_latency_ms = profile.method_execute_ms
        summary.runtime_delegation = profile.runtime_delegation_fraction
        # The profiler's own accumulated portable event time - not a p50
        # multiplied by a percentage, which would mix two measurements.
        summary.portable_runtime_ms = profile.portable_ms
        kernels = sorted(profile.portable_kernels,
                         key=lambda kernel: kernel.total_ms, reverse=True)
        summary.portable_operator_count = len(kernels)

        portable_total = profile.portable_ms or 0.0
        views = [
            HotspotView(
                operator=kernel.operator_name,
                runtime_ms=kernel.total_ms,
                total_fraction=kernel.runtime_fraction,
                # Guarded: a profile with no portable time has no share to take.
                portable_fraction=(kernel.total_ms / portable_total
                                   if portable_total > 0 else 0.0),
            )
            for kernel in kernels
        ]
        if views:
            chosen = 0
            if focus_kernel:
                chosen = next((index for index, kernel in enumerate(kernels)
                               if kernel.name == focus_kernel), 0)
            summary.top_hotspot = views[chosen]
            summary.other_hotspots = [view for index, view in enumerate(views)
                                      if index != chosen][:OTHER_HOTSPOT_LIMIT]

    if configuration is not None:
        summary.provider_label = configuration.definition.label
        summary.provider_model = configuration.model
        summary.provider_is_local = configuration.is_local

    return summary


# --- rendering ---------------------------------------------------------------


def _percent(fraction: Optional[float]) -> str:
    return "—" if fraction is None else f"{100 * fraction:.1f}%"


def format_decision_screen(summary: RepairOpportunitySummary) -> str:
    """The screen shown before asking whether to spend a provider request.

    Ordered so the question "is this worth it?" can be answered top to bottom:
    how healthy the deployment is, how much runs outside the delegate, what the
    single worst offender costs, and what perfect removal could buy.
    """
    lines = ["DelegateDoctor AI Repair Exploration", "",
             "No known DelegateDoctor repairs remain.", ""]

    lines.append("Baseline on target")
    if summary.target:
        lines.append(f"  Target                  {summary.target}")
    if summary.measured_latency_ms is not None:
        lines.append(f"  Method::execute         "
                     f"{summary.measured_latency_ms:.3f} ms (profiled)")
    if summary.operator_delegation is not None:
        lines.append(f"  Operator delegation     "
                     f"{_percent(summary.operator_delegation)}")
    if summary.runtime_delegation is not None:
        lines.append(f"  Runtime delegation      "
                     f"{_percent(summary.runtime_delegation)}")

    if summary.portable_runtime_fraction is not None:
        lines += ["", "Portable execution",
                  f"  Runtime share           "
                  f"{_percent(summary.portable_runtime_fraction)}"]
        if summary.portable_runtime_ms is not None:
            lines.append(f"  Measured event time     "
                         f"{summary.portable_runtime_ms:.3f} ms")
        lines.append(f"  Portable operators      "
                     f"{summary.portable_operator_count}")

    hotspot = summary.top_hotspot
    if hotspot is not None:
        lines += ["", "Top hotspot",
                  f"  Operator                {hotspot.operator}",
                  f"  Runtime                 {hotspot.runtime_ms:.3f} ms",
                  f"  Share of total runtime  {_percent(hotspot.total_fraction)}",
                  f"  Share of fallback       "
                  f"{_percent(hotspot.portable_fraction)}"]

    if summary.other_hotspots:
        lines += ["", "Other portable hotspots"]
        for other in summary.other_hotspots:
            lines.append(f"  {other.operator:<22}  {other.runtime_ms:.3f} ms · "
                         f"{_percent(other.total_fraction)}")

    # No theoretical bound here. The consent screen answers "may I send this?",
    # and the measured shares above already say how much is at stake; the bound
    # is kept in report.txt and report.html, where there is room to read it.
    lines += ["", f"Catalog repair            {summary.catalog_match}"]

    if summary.provider_label:
        if summary.provider_is_local:
            lines += [f"Provider                  {summary.provider_label}",
                      "Processing                Local - nothing leaves this "
                      "machine"]
        else:
            lines.append(f"Provider                  {summary.provider_label} · "
                         f"{summary.provider_model}")

    lines += ["", _privacy_block(summary)]
    return "\n".join(lines)


def _privacy_block(summary: RepairOpportunitySummary) -> str:
    """Exactly what a *repair* request contains - not what preparation sends.

    Kept as two explicit lists rather than a summary sentence. This is the
    screen someone answers "y" on, and "a sanitized neighbourhood of the graph"
    does not tell them whether their weights are in it. The second list is the
    one that matters: it is the only place the tool commits, in front of the
    user, to what never leaves the machine.
    """
    destination = ("your local provider" if summary.provider_is_local
                   else "your configured AI provider")
    return (
        f"DelegateDoctor can experimentally inspect the measured exported\n"
        f"graph for additional model-specific repairs, by sending a sanitized\n"
        f"description of it to {destination}.\n"
        f"\n"
        f"It will send:      operator names, graph relationships, tensor\n"
        f"                   shapes and dtypes, profiling metadata\n"
        f"It will NOT send:  your model source, model weights, tensor values,\n"
        f"                   representative inputs, checkpoints, API keys\n"
    )


def format_report_section(summary: RepairOpportunitySummary) -> str:
    """The same facts for report.txt, kept whether or not AI ever ran."""
    lines = ["", "REPAIR OPPORTUNITY", "-" * 40]
    if summary.measured_latency_ms is not None:
        lines.append(f"Method::execute         {summary.measured_latency_ms:.3f} ms")
    if summary.operator_delegation is not None:
        lines.append(f"Operator delegation     "
                     f"{_percent(summary.operator_delegation)}")
    if summary.runtime_delegation is not None:
        lines.append(f"Runtime delegation      "
                     f"{_percent(summary.runtime_delegation)}")
    if summary.portable_runtime_fraction is not None:
        lines.append(f"Portable runtime        "
                     f"{_percent(summary.portable_runtime_fraction)}")
    if summary.portable_runtime_ms is not None:
        lines.append(f"Portable event time     "
                     f"{summary.portable_runtime_ms:.3f} ms")

    hotspot = summary.top_hotspot
    if hotspot is not None:
        lines.append(f"Top hotspot             {hotspot.operator} · "
                     f"{hotspot.runtime_ms:.3f} ms · "
                     f"{_percent(hotspot.total_fraction)} of runtime · "
                     f"{_percent(hotspot.portable_fraction)} of fallback")
    for other in summary.other_hotspots:
        lines.append(f"  also                  {other.describe()}")

    ceiling = summary.theoretical_upper_bound_speedup
    if ceiling is not None:
        lines.append(f"Theoretical upper bound {ceiling:.2f}x")

    lines.append(f"Catalog repair          {summary.catalog_match}")
    # Only when experimental AI repair actually figured in the run. Printing
    # "Not requested" on every default run advertises an opt-in feature as a
    # step that did not happen.
    if summary.ai_status != AI_NOT_REQUESTED:
        lines.append(f"AI exploration          {summary.ai_status}")
    return "\n".join(lines)
