"""Render one `OptimizationResult` as a self-contained `report.html`.

This layer only formats. It computes nothing, re-runs nothing and re-measures
nothing: every number here already exists in the result object or in the run's
JSON artifacts. If a value is absent it is shown as "not measured" rather than
as a zero.

The file is deliberately portable - all CSS inline, no JavaScript, no fonts, no
images, no network of any kind - so `file:///.../report.html` is enough, and the
report survives being emailed to a colleague.

The report runs on the developer's machine. The Arm64 Android target only
executes, profiles, verifies and benchmarks; nothing is ever displayed there.
"""

from __future__ import annotations

import html
import os
from typing import Optional

from . import result as result_module

REPORT_FILENAME = "report.html"

# Restrained semantic palette. Every colour is paired with a text label, so the
# report is still readable in greyscale or with colour-vision deficiency.
CSS = """
:root {
  --bg: #f6f7f9;
  --surface: #ffffff;
  --ink: #12151a;
  --ink-soft: #565d69;
  --ink-mute: #7b8492;
  --line: #e3e6ea;
  --line-strong: #cfd4db;
  --accent: #2f3f8f;
  --ok: #1f7a4d;
  --ok-soft: #e6f3ec;
  --warn: #9a6512;
  --warn-soft: #fdf2df;
  --bad: #a52a2a;
  --bad-soft: #fbecec;
  --idle: #6b7280;
  --idle-soft: #eef0f3;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas,
          "Liberation Mono", monospace;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  padding: 40px 24px 80px;
  background: var(--bg);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               Helvetica, Arial, sans-serif;
  font-size: 15px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

.page { max-width: 940px; margin: 0 auto; }

/* --- masthead --------------------------------------------------------- */

.masthead {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 24px;
  flex-wrap: wrap;
  padding-bottom: 20px;
  border-bottom: 2px solid var(--ink);
}
.wordmark {
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
}
.model-name {
  margin: 6px 0 2px;
  font-size: 30px;
  font-weight: 640;
  letter-spacing: -0.02em;
}
.backend { color: var(--ink-mute); font-size: 13px; }

.verdict {
  padding: 8px 14px;
  border-radius: 4px;
  border: 1px solid;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.08em;
  white-space: nowrap;
}
.verdict.ok   { color: var(--ok);   background: var(--ok-soft);   border-color: #b7ddc7; }
.verdict.warn { color: var(--warn); background: var(--warn-soft); border-color: #e8d3a6; }
.verdict.bad  { color: var(--bad);  background: var(--bad-soft);  border-color: #eec4c4; }
.verdict.idle { color: var(--idle); background: var(--idle-soft); border-color: var(--line-strong); }

/* --- sections --------------------------------------------------------- */

section { margin-top: 34px; }

h2 {
  margin: 0 0 14px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--ink-mute);
}

.card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 20px 22px;
}

.grid-2 {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 16px;
}

/* --- hero ------------------------------------------------------------- */

.hero {
  background: var(--surface);
  border: 1px solid var(--line);
  border-left: 4px solid var(--accent);
  border-radius: 6px;
  padding: 28px 26px;
  text-align: center;
}
.hero.ok  { border-left-color: var(--ok); }
.hero.bad { border-left-color: var(--bad); }
.hero-figure {
  font-size: 56px;
  font-weight: 680;
  letter-spacing: -0.03em;
  line-height: 1.05;
  font-variant-numeric: tabular-nums;
}
.hero-figure.ok  { color: var(--ok); }
.hero-figure.bad { color: var(--bad); }
.hero-label {
  margin-top: 4px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--ink-mute);
}
.hero-detail {
  margin-top: 18px;
  padding-top: 16px;
  border-top: 1px solid var(--line);
  font-family: var(--mono);
  font-size: 15px;
  color: var(--ink-soft);
}
.hero-note { margin-top: 10px; font-size: 13px; color: var(--ink-mute); }

/* --- metrics ---------------------------------------------------------- */

.metric-label {
  font-size: 12px;
  color: var(--ink-mute);
  letter-spacing: 0.04em;
}
.metric-value {
  font-size: 28px;
  font-weight: 640;
  letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums;
}
.metric-value.small { font-size: 20px; }

/* --- bars ------------------------------------------------------------- */

.bar {
  position: relative;
  height: 10px;
  margin-top: 10px;
  background: #eceef1;
  border-radius: 5px;
  overflow: hidden;
}
.bar > span {
  display: block;
  height: 100%;
  border-radius: 5px;
  background: var(--accent);
}
.bar.ok   > span { background: var(--ok); }
.bar.warn > span { background: var(--warn); }
.bar.bad  > span { background: var(--bad); }

.compare { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.arrow {
  color: var(--ink-mute);
  font-family: var(--mono);
  padding: 0 6px;
}

/* --- rows ------------------------------------------------------------- */

.rows { width: 100%; border-collapse: collapse; }
.rows td {
  padding: 9px 0;
  border-bottom: 1px solid var(--line);
  vertical-align: baseline;
}
.rows tr:last-child td { border-bottom: none; }
.rows td.key { color: var(--ink-soft); }
.rows td.val {
  text-align: right;
  font-family: var(--mono);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.rows td.val.wide { white-space: normal; text-align: right; }

.tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 3px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.08em;
  font-family: inherit;
}
.tag.ok   { color: var(--ok);   background: var(--ok-soft); }
.tag.warn { color: var(--warn); background: var(--warn-soft); }
.tag.bad  { color: var(--bad);  background: var(--bad-soft); }
.tag.idle { color: var(--idle); background: var(--idle-soft); }

code, .mono { font-family: var(--mono); font-size: 13px; }

/* --- hotspots --------------------------------------------------------- */

.hotspot { padding: 12px 0; border-bottom: 1px solid var(--line); }
.hotspot:last-child { border-bottom: none; }
.hotspot-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 16px;
}
.hotspot-name { font-family: var(--mono); font-size: 14px; }
.hotspot-cost {
  font-family: var(--mono);
  font-variant-numeric: tabular-nums;
  color: var(--ink-soft);
  white-space: nowrap;
}
.more { margin-top: 12px; font-size: 13px; color: var(--ink-mute); }

/* --- repair diagram --------------------------------------------------- */

.flow { display: flex; gap: 28px; flex-wrap: wrap; }
.flow-col { flex: 1; min-width: 180px; }
.flow-title {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.14em;
  color: var(--ink-mute);
  margin-bottom: 10px;
}
.node {
  border: 1px solid var(--line-strong);
  border-radius: 4px;
  padding: 6px 10px;
  background: #fbfbfc;
  font-family: var(--mono);
  font-size: 12px;
  text-align: center;
}
.node.bad { border-color: #eec4c4; background: var(--bad-soft); color: var(--bad); }
.node.ok  { border-color: #b7ddc7; background: var(--ok-soft);  color: var(--ok); }
.link {
  width: 1px;
  height: 12px;
  margin: 0 auto;
  background: var(--line-strong);
}

/* --- pipeline strip --------------------------------------------------- */

.pipeline {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(104px, 1fr));
  gap: 1px;
  background: var(--line);
  border: 1px solid var(--line);
  border-radius: 6px;
  overflow: hidden;
}
.stage { background: var(--surface); padding: 12px 10px; text-align: center; }
.stage-name {
  font-size: 11px;
  color: var(--ink-mute);
  margin-bottom: 5px;
  letter-spacing: 0.03em;
}
.stage-status { font-size: 11px; font-weight: 700; letter-spacing: 0.06em; }
.stage-status.ok   { color: var(--ok); }
.stage-status.warn { color: var(--warn); }
.stage-status.bad  { color: var(--bad); }
.stage-status.idle { color: var(--idle); }

/* --- details ---------------------------------------------------------- */

details {
  margin-top: 34px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 6px;
}
summary {
  padding: 14px 22px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-soft);
}
summary::-webkit-details-marker { color: var(--ink-mute); }
.details-body { padding: 4px 22px 22px; }
.details-body h3 {
  margin: 22px 0 8px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-mute);
}

.note {
  margin-top: 14px;
  padding: 12px 14px;
  border-left: 3px solid var(--line-strong);
  background: #fafbfc;
  font-size: 13px;
  color: var(--ink-soft);
}
.note.warn { border-left-color: var(--warn); background: var(--warn-soft); }

footer {
  margin-top: 40px;
  padding-top: 18px;
  border-top: 1px solid var(--line);
  font-size: 12px;
  color: var(--ink-mute);
}
"""

# --- small formatting helpers ----------------------------------------------

NOT_MEASURED = '<span class="tag idle">NOT MEASURED</span>'


def esc(value) -> str:
    """Escape anything before it reaches the page.

    Model names, operator names, device strings, exception text and paths are
    all attacker-adjacent at worst and typo-adjacent at best. Nothing reaches
    the HTML without passing through here.
    """
    return html.escape("" if value is None else str(value), quote=True)


def _percent(fraction: Optional[float]) -> str:
    return "—" if fraction is None else f"{100 * fraction:.1f}%"


def _ms(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.2f} ms"


def _width(fraction: Optional[float]) -> str:
    """Bar width, clamped. Bars always start at zero and never exceed 100%."""
    if fraction is None:
        return "0"
    return f"{max(0.0, min(1.0, fraction)) * 100:.1f}"


def _health(fraction: Optional[float]) -> str:
    """Colour class for a delegation fraction, paired with text elsewhere."""
    if fraction is None:
        return "idle"
    if fraction >= 0.95:
        return "ok"
    if fraction >= 0.75:
        return "warn"
    return "bad"


STATUS_TONE = {
    result_module.PASS: "ok",
    result_module.NONE_FOUND: "ok",
    result_module.FAILED: "bad",
    result_module.UNSUPPORTED: "warn",
    result_module.UNAVAILABLE: "idle",
    result_module.NOT_RUN: "idle",
}

# Short stage captions for the strip; the full names stay in the details.
STAGE_SHORT = {
    result_module.EXPORT: "Export",
    result_module.GRAPH: "Graph",
    result_module.LOWERING: "Lowering",
    result_module.DELEGATION: "XNNPACK",
    result_module.DEVICE: "Device",
    result_module.PROFILING: "Profile",
    result_module.REPAIR: "Repair",
    result_module.VERIFICATION: "Verify",
    result_module.BENCHMARK: "Benchmark",
}

# The headline shown top-right, and its tone.
VERDICT = {
    result_module.REPAIR_ACCEPTED: ("REPAIR ACCEPTED", "ok"),
    result_module.REPAIR_REJECTED: ("REPAIR REJECTED", "bad"),
    result_module.FULLY_DELEGATED: ("DEPLOYMENT HEALTHY", "ok"),
    result_module.NO_REPAIR_REQUIRED: ("DEPLOYMENT HEALTHY", "ok"),
    result_module.NO_REPAIR_AVAILABLE: ("NO REPAIR AVAILABLE", "warn"),
    result_module.ANALYSIS_COMPLETE: ("STATIC ANALYSIS COMPLETE", "idle"),
    result_module.DEVICE_EXECUTION_UNSUPPORTED: ("STATIC ANALYSIS COMPLETE", "idle"),
    result_module.EXECUTORCH_LOWERING_UNSUPPORTED: ("LOWERING UNSUPPORTED", "bad"),
}


def _row(key: str, value: str, tone: str = "", wide: bool = False) -> str:
    css = "val wide" if wide else "val"
    body = f'<span class="tag {tone}">{value}</span>' if tone else value
    return f'<tr><td class="key">{key}</td><td class="{css}">{body}</td></tr>'


def _bar(fraction: Optional[float], tone: str = "") -> str:
    return (f'<div class="bar {tone}"><span style="width:{_width(fraction)}%">'
            f'</span></div>')


# --- sections ---------------------------------------------------------------


def _masthead(outcome) -> str:
    label, tone = VERDICT.get(outcome.status, (outcome.status.replace("_", " "), "idle"))
    name = esc(outcome.model_name or "PyTorch Model")
    return (
        f'<header class="masthead">'
        f'<div>'
        f'<div class="wordmark">DelegateDoctor</div>'
        f'<h1 class="model-name">{name}</h1>'
        f'<div class="backend">ExecuTorch + XNNPACK</div>'
        f'</div>'
        f'<div class="verdict {tone}">{esc(label)}</div>'
        f'</header>'
    )


def _target(outcome) -> str:
    """Which Arm target produced the measurements, or that none did."""
    if not outcome.device_description:
        stage = outcome.stage(result_module.DEVICE)
        reason = esc(stage.detail) if stage and stage.detail else "No Arm64 target attached."
        return (
            '<section><h2>Target</h2><div class="card">'
            '<div class="metric-value small">No measurement target</div>'
            f'<div class="note">{reason}<br>'
            'Static analysis below was produced on this machine. No runtime '
            'numbers are reported.</div>'
            '</div></section>'
        )

    kind = "Measured on emulator" if outcome.device_is_emulator else "Measured on device"
    caveat = ""
    if outcome.device_is_emulator:
        caveat = ('<div class="note warn">Emulator numbers are not handset '
                  'numbers. Cache sizes, memory bandwidth and scheduling differ '
                  'from a physical phone.</div>')
    return (
        '<section><h2>Target</h2><div class="card">'
        f'<div class="metric-value small mono">{esc(outcome.device_description)}</div>'
        f'<div class="metric-label" style="margin-top:6px">{kind}</div>'
        f'{caveat}'
        '</div></section>'
    )


def _hero(outcome) -> str:
    """The one number a reader should see first."""
    benchmark = outcome.benchmark

    if benchmark is not None:
        speedup = benchmark.p50_speedup
        faster = speedup >= 1.0
        tone = "ok" if outcome.status == result_module.REPAIR_ACCEPTED else "bad"
        if faster:
            figure, label = f"{speedup:.2f}x", "FASTER"
        else:
            slower = (benchmark.after.p50_ms / benchmark.before.p50_ms - 1.0) * 100
            figure, label = f"{slower:.1f}%", "SLOWER"
        note = ""
        if outcome.status == result_module.REPAIR_REJECTED and outcome.decision:
            note = f'<div class="hero-note">{esc(outcome.decision.headline)}</div>'
        return (
            f'<section><div class="hero {tone}">'
            f'<div class="hero-figure {tone}">{figure}</div>'
            f'<div class="hero-label">{label}</div>'
            f'<div class="hero-detail">'
            f'{_ms(benchmark.before.p50_ms)} &rarr; {_ms(benchmark.after.p50_ms)}'
            f'</div>'
            f'<div class="hero-note">median of {benchmark.repetitions} '
            f'interleaved repetition(s), {benchmark.measured_iterations} measured '
            f'iterations each, {benchmark.threads} threads</div>'
            f'{note}'
            f'</div></section>'
        )

    # No benchmark: lead with whatever the run actually established.
    if outcome.status in (result_module.FULLY_DELEGATED,
                          result_module.NO_REPAIR_REQUIRED):
        fraction = (outcome.before_profile.runtime_delegation_fraction
                    if outcome.before_profile is not None
                    else outcome.before_delegation.operator_delegation_fraction
                    if outcome.before_delegation is not None else None)
        return (
            f'<section><div class="hero ok">'
            f'<div class="hero-figure ok">{_percent(fraction)}</div>'
            f'<div class="hero-label">Delegated to XNNPACK</div>'
            f'<div class="hero-detail">No portable hotspot detected</div>'
            f'<div class="hero-note">DelegateDoctor analyzed this model and '
            f'found no meaningful fallback to repair.</div>'
            f'</div></section>'
        )

    if outcome.before_delegation is not None:
        fraction = outcome.before_delegation.operator_delegation_fraction
        runtime = (outcome.before_profile.runtime_delegation_fraction
                   if outcome.before_profile is not None else None)
        headline = _percent(runtime) if runtime is not None else _percent(fraction)
        caption = ("Runtime delegation" if runtime is not None
                   else "Operator-count delegation")
        return (
            f'<section><div class="hero">'
            f'<div class="hero-figure">{headline}</div>'
            f'<div class="hero-label">{caption}</div>'
            f'<div class="hero-detail">'
            f'{outcome.before_delegation.portable_op_total} portable of '
            f'{outcome.before_delegation.total_ops} operators</div>'
            f'</div></section>'
        )

    label, tone = VERDICT.get(outcome.status, (outcome.status, "idle"))
    return (
        f'<section><div class="hero {tone}">'
        f'<div class="hero-figure {tone}" style="font-size:32px">{esc(label)}</div>'
        f'<div class="hero-note">{esc(outcome.summary)}</div>'
        f'</div></section>'
    )


def _delegation(outcome) -> str:
    """Operator-count delegation beside runtime-weighted delegation.

    The project's central finding: these two can disagree badly, and when they
    do the operator count is the misleading one.
    """
    before = outcome.before_delegation
    if before is None:
        return ""

    operator = before.operator_delegation_fraction
    runtime = (outcome.before_profile.runtime_delegation_fraction
               if outcome.before_profile is not None else None)

    runtime_cell = (
        f'<div class="metric-value">{_percent(runtime)}</div>'
        + _bar(runtime, _health(runtime))
        if runtime is not None else
        f'<div class="metric-value small">{NOT_MEASURED}</div>'
        '<div class="metric-label" style="margin-top:8px">'
        'Requires profiling on the Arm target</div>'
    )

    if runtime is None:
        interpretation = ("Runtime weighting was not measured, so operator "
                          "count is the only delegation signal available here.")
    elif operator - runtime > 0.10:
        interpretation = ("A small number of portable operations dominate "
                          "execution time. Operator-count delegation "
                          "understates the cost of the remaining fallbacks.")
    elif runtime >= 0.95:
        interpretation = "No significant portable runtime bottleneck detected."
    else:
        interpretation = ("Operator-count and runtime-weighted delegation "
                          "broadly agree for this model.")

    return (
        '<section><h2>Delegation health</h2><div class="card">'
        '<div class="compare">'
        '<div>'
        '<div class="metric-label">Operator-count delegation</div>'
        f'<div class="metric-value">{_percent(operator)}</div>'
        f'{_bar(operator, _health(operator))}'
        '</div>'
        '<div>'
        '<div class="metric-label">Runtime-weighted delegation</div>'
        f'{runtime_cell}'
        '</div>'
        '</div>'
        f'<div class="note">{interpretation}</div>'
        '</div></section>'
    )


def _hotspots(outcome, limit: int = 3) -> str:
    profile = outcome.before_profile
    if profile is None:
        return ""

    if not profile.portable_kernels:
        return (
            '<section><h2>Portable hotspots</h2><div class="card">'
            '<div class="metric-value small">None</div>'
            '<div class="note">All measured runtime is inside XNNPACK.</div>'
            '</div></section>'
        )

    repairable = {}
    for rule_id, found in outcome.detections.items():
        if not found.applies:
            continue
        for kernel in profile.portable_kernels:
            if kernel.name in repairable:
                continue
            # The pipeline already matched these; recompute cheaply by name.
            if rule_id in outcome.repair_catalog and _kernel_matches(
                    outcome, rule_id, kernel.name):
                repairable[kernel.name] = rule_id

    rows = []
    for kernel in profile.portable_kernels[:limit]:
        rule_id = repairable.get(kernel.name)
        badge = (f'<span class="tag ok">{esc(rule_id)}</span>' if rule_id
                 else '<span class="tag idle">NO RULE</span>')
        rows.append(
            f'<div class="hotspot">'
            f'<div class="hotspot-head">'
            f'<span class="hotspot-name">{esc(kernel.operator_name)}</span>'
            f'<span class="hotspot-cost">{kernel.total_ms:.2f} ms &middot; '
            f'{_percent(kernel.runtime_fraction)} {badge}</span>'
            f'</div>'
            f'{_bar(kernel.runtime_fraction, "warn" if not rule_id else "")}'
            f'</div>'
        )

    remaining = len(profile.portable_kernels) - limit
    more = (f'<div class="more">+ {remaining} additional portable operator(s), '
            f'listed under Technical details</div>' if remaining > 0 else "")

    return ('<section><h2>Portable hotspots</h2><div class="card">'
            + "".join(rows) + more + '</div></section>')


def _repair_opportunity(outcome) -> str:
    """What a repair could be worth, from the same object the terminal used.

    Nothing here is recomputed: every number comes from the summary the
    pipeline built once, so the report cannot quietly disagree with the screen
    the user answered a question on.
    """
    summary = getattr(outcome, "opportunity", None)
    if summary is None or not summary.has_measurement:
        return ""

    rows = []
    if summary.measured_latency_ms is not None:
        rows.append(_row("Method::execute",
                         f"{summary.measured_latency_ms:.3f} ms"))
    if summary.operator_delegation is not None:
        rows.append(_row("Operator delegation",
                         _percent(summary.operator_delegation)))
    rows.append(_row("Runtime delegation", _percent(summary.runtime_delegation)))
    rows.append(_row("Portable runtime",
                     _percent(summary.portable_runtime_fraction)))
    if summary.portable_runtime_ms is not None:
        rows.append(_row("Portable event time",
                         f"{summary.portable_runtime_ms:.3f} ms"))
    rows.append(_row("Catalog repair", esc(summary.catalog_match)))
    rows.append(_row("AI exploration", esc(summary.ai_status)))

    hotspot = summary.top_hotspot
    lead = ""
    if hotspot is not None:
        lead = (
            f'<div class="metric-value small">{esc(hotspot.operator)}</div>'
            f'<div class="note">{hotspot.runtime_ms:.3f} ms &middot; '
            f'{_percent(hotspot.total_fraction)} of measured runtime &middot; '
            f'{_percent(hotspot.portable_fraction)} of all fallback</div>'
            f'{_bar(hotspot.total_fraction, "warn")}'
        )

    ceiling = summary.theoretical_upper_bound_speedup
    bound = ""
    if ceiling is not None:
        # A number in a labelled row, like every other measurement here. The
        # word "theoretical" carries the caveat; a paragraph repeating it on
        # every run does not earn its space.
        rows.insert(0, _row("Theoretical upper bound", f"{ceiling:.2f}x"))

    others = ""
    if summary.other_hotspots:
        items = "".join(
            f'<div class="more">{esc(other.operator)} &middot; '
            f'{other.runtime_ms:.3f} ms &middot; '
            f'{_percent(other.total_fraction)}</div>'
            for other in summary.other_hotspots)
        others = f'<div class="note">Other portable operators</div>{items}'

    return ('<section><h2>Repair opportunity</h2><div class="card">'
            + lead + bound
            + f'<table class="rows">{"".join(rows)}</table>'
            + others + '</div></section>')


def _share_line(attempt) -> str:
    """The measured share a catalog attempt targeted, when there was one.

    A model-level AI candidate has no single share: it is a proposal about the
    graph, and putting an operator percentage on it would imply a targeting
    decision DelegateDoctor did not make.
    """
    if attempt.hotspot is None:
        return f'<div class="note">{esc(attempt.source)}</div>'
    tone = {"ACCEPTED": "ok", "REJECTED": "warn"}.get(attempt.status, "idle")
    # A catalog rule is applied to every site it matches, so the share it
    # represents is the sites' total - not the one site that happened to rank
    # highest. Falling back keeps AI rows, which have no site set, unchanged.
    share = (attempt.represented_runtime if attempt.represented_runtime is not None
             else attempt.hotspot.runtime_share)
    sites = ""
    if attempt.matching_sites:
        sites = (f' &middot; {attempt.matching_sites} matching site'
                 f'{"" if attempt.matching_sites == 1 else "s"}')
    return (f'<div class="note">runtime share before &middot; '
            f'{_percent(share)}{sites} &middot; '
            f'{esc(attempt.source)}</div>'
            f'{_bar(share, tone)}')


def _journey(outcome) -> str:
    """The optimization sequence, made visually obvious.

    A single before/after table would hide the thing that matters most about
    this run: that it was iterative, that each step was measured against the
    one before it, and that some steps were rejected. So each attempt gets its
    own row, in order, including the ones that did not survive.
    """
    history = getattr(outcome, "repair_history", None)
    if history is None or not history.attempts:
        return ""

    def endpoint(title, operator_delegation, runtime_delegation, latency):
        rows = []
        if operator_delegation is not None:
            rows.append(_row("Operator delegation", _percent(operator_delegation)))
        if runtime_delegation is not None:
            rows.append(_row("Runtime delegation", _percent(runtime_delegation)))
        if latency is not None:
            rows.append(_row("p50", _ms(latency)))
        if not rows:
            return ""
        return (f'<div class="card"><div class="note">{esc(title)}</div>'
                f'<table class="rows">{"".join(rows)}</table></div>')

    steps = []
    for position, attempt in enumerate(history.attempts, start=1):
        tone = {"ACCEPTED": "ok", "REJECTED": "warn"}.get(attempt.status, "idle")
        latency = ""
        if attempt.before_latency_ms and attempt.after_latency_ms:
            latency = (f'<div class="note">p50 '
                       f'{_ms(attempt.before_latency_ms)} &rarr; '
                       f'{_ms(attempt.after_latency_ms)}'
                       + (f' &middot; {attempt.speedup:.2f}x'
                          if attempt.speedup else "")
                       + '</div>')
        gates = []
        if attempt.host_verification_passed is not None:
            gates.append("host " + ("PASS" if attempt.host_verification_passed
                                    else "FAIL"))
        if attempt.device_verification_passed is not None:
            gates.append("device " + ("PASS" if attempt.device_verification_passed
                                      else "FAIL"))
        # Named separately from the two correctness gates, because it is a
        # statement about the backend rather than about this repair.
        if attempt.backend_fidelity and attempt.backend_fidelity != "OK":
            gates.append(f"backend fidelity {attempt.backend_fidelity}")
        gate_line = (f'<div class="note">{" &middot; ".join(esc(gate) for gate in gates)}</div>'
                     if gates else "")
        reason = (f'<div class="note">{esc(attempt.reason)}</div>'
                  if attempt.reason else "")

        runtimes = ""

        steps.append(
            f'<div class="hotspot">'
            f'<div class="hotspot-head">'
            f'<span class="hotspot-name">{position}. '
            f'{esc(attempt.subject)}</span>'
            f'<span class="hotspot-cost">{esc(attempt.label)} '
            f'<span class="tag {tone}">{esc(attempt.status)}</span></span>'
            f'</div>'
            f'{_share_line(attempt)}'
            f'{runtimes}{gate_line}{latency}{reason}'
            f'</div>'
        )

    totals = ""
    if history.total_speedup:
        totals = (f'<div class="card"><div class="metric-value">'
                  f'{history.total_speedup:.2f}x</div>'
                  f'<div class="note">original to final, measured end to end on '
                  f'the same target</div></div>')

    stop = (f'<div class="note">Stopped because {esc(history.stop_reason)}</div>'
            if history.stop_reason else "")

    policy = _journey_policy(history)

    return (
        '<section><h2>Optimization journey</h2>'
        + policy
        + endpoint("Original", history.original_operator_delegation,
                   history.original_runtime_delegation,
                   history.original_latency_ms)
        + f'<div class="card">{"".join(steps)}{stop}</div>'
        + endpoint("Final", history.final_operator_delegation,
                   history.final_runtime_delegation,
                   history.final_latency_ms)
        + totals
        + '</section>'
    )


# What the two thresholds and the single consent decision mean, said once at
# the top of the journey rather than repeated against every step.
_AI_CONSENT_TEXT = {
    "granted": ("approved once for this run, covering every remaining "
                "hotspot"),
    "declined": "declined, so no AI repair was attempted",
    "unavailable": "no AI provider was configured, so none was attempted",
    "not needed": "not required: no eligible unknown hotspot came up",
}


def _journey_policy(history) -> str:
    """How this run chose what to repair, in the order it actually applied."""
    from . import model_exploration, repair_loop

    consent = _AI_CONSENT_TEXT.get(history.ai_consent, esc(history.ai_consent))
    offered = ""
    if history.ai_hotspots_offered:
        offered = (f' &middot; {history.ai_hotspots_offered} hotspot(s) '
                   f'presented')

    rows = [
        _row("Known repairs",
             f"applied automatically above "
             f"{100 * repair_loop.MIN_DD_HOTSPOT_RUNTIME_SHARE:.1f}% of runtime"),
        _row("AI repairs",
             f"one bounded investigation of the whole model, offered when "
             f"portable runtime exceeds "
             f"{100 * model_exploration.MIN_AI_PORTABLE_RUNTIME_SHARE:.0f}%",
             wide=True),
        _row("AI consent", esc(consent) + offered, wide=True),
    ]
    return ('<div class="card"><table class="rows">'
            + "".join(rows)
            + '</table><div class="note">A known repair is tried before any AI '
              'proposal, at every re-profile. A hotspot the catalog recognises '
              'is never sent to a provider.</div></div>')


def _kernel_matches(outcome, rule_id: str, kernel_name: str) -> bool:
    """Ask the catalog, without importing rule internals into the report."""
    matcher = outcome.repair_catalog.get(rule_id, {}).get("matches")
    try:
        return bool(matcher and matcher(kernel_name))
    except Exception:
        return False


def _repair(outcome) -> str:
    matched = [rule_id for rule_id, found in outcome.detections.items()
               if found.applies]
    if not matched:
        if outcome.status in (result_module.FULLY_DELEGATED,
                              result_module.NO_REPAIR_REQUIRED):
            body = ('<div class="metric-value small">Not required</div>'
                    '<div class="note">No portable fallback worth repairing was '
                    'found in this graph.</div>')
        else:
            body = ('<div class="metric-value small">No matching repair</div>'
                    '<div class="note">The hotspots above are real, but no rule '
                    'in the catalog recognises them. They are candidates for a '
                    'future repair rule; nothing was changed.</div>')
        return f'<section><h2>Repair</h2><div class="card">{body}</div></section>'

    blocks = []
    for rule_id in matched:
        meta = outcome.repair_catalog.get(rule_id, {})
        found = outcome.detections[rule_id]
        sites = found.detections
        problem = esc(sites[0].explain().splitlines()[0]) if sites else ""
        applied = outcome.repairs_applied.get(rule_id)

        rows = [_row("Problem", f'<span class="mono">{problem}</span>', wide=True)]
        if meta.get("rewrite"):
            rows.append(_row("Repair", esc(meta["rewrite"]), wide=True))
        rows.append(_row("Sites found", str(len(sites))))
        if applied is not None:
            rows.append(_row("Sites repaired", str(applied)))
        else:
            rows.append(_row("Applied", "NO", "idle"))

        blocks.append(
            f'<div class="card" style="margin-bottom:14px">'
            f'<div class="metric-value small">'
            f'<span class="mono">{esc(rule_id)}</span></div>'
            f'<div class="metric-label" style="margin-bottom:10px">'
            f'{esc(meta.get("title", ""))}</div>'
            f'<table class="rows">{"".join(rows)}</table>'
            f'{_repair_flow(outcome, rule_id, meta)}'
            f'</div>'
        )

    if not outcome.repairs_applied:
        blocks.append(
            '<div class="note">Pattern recognised but not applied. A repair is '
            'only accepted once it verifies and benchmarks faster on the Arm '
            'target, and those stages did not run.</div>'
        )
    return '<section><h2>Repair</h2>' + "".join(blocks) + '</section>'


def _repair_flow(outcome, rule_id: str, meta: dict) -> str:
    """A small before/after structure diagram, when the rule described one.

    Generic on purpose: the rule metadata supplies the node lists, so nothing
    here is specific to DD-001 or any other rule.
    """
    before = meta.get("flow_before")
    after = meta.get("flow_after")
    if not before or not after:
        return ""

    def column(title: str, nodes, terminal: str, tone: str) -> str:
        parts = [f'<div class="flow-title">{esc(title)}</div>']
        for index, node in enumerate(nodes):
            if index:
                parts.append('<div class="link"></div>')
            parts.append(f'<div class="node">{esc(node)}</div>')
        parts.append('<div class="link"></div>')
        parts.append(f'<div class="node {tone}">{esc(terminal)}</div>')
        return f'<div class="flow-col">{"".join(parts)}</div>'

    return (
        '<div class="flow" style="margin-top:18px">'
        + column("Before", before, "portable execution", "bad")
        + column("After", after, "XNNPACK delegate", "ok")
        + '</div>'
    )


def _correctness(outcome) -> str:
    host = outcome.host_verification
    device = outcome.device_verification
    if host is None and device is None:
        stage = outcome.stage(result_module.VERIFICATION)
        if stage and stage.status == result_module.UNSUPPORTED:
            return ('<section><h2>Correctness</h2><div class="card">'
                    '<div class="metric-value small">'
                    '<span class="tag warn">NOT VERIFIABLE</span></div>'
                    f'<div class="note">{esc(stage.detail)}<br>'
                    'A repair DelegateDoctor cannot verify is never accepted.'
                    '</div></div></section>')
        return ""

    rows = []
    if host is not None:
        rows.append(_row("Host verification", host.status_text,
                         "ok" if host.passed else "bad"))
    if device is not None:
        rows.append(_row("Android verification", device.status_text,
                         "ok" if device.passed else "bad"))

    metrics = getattr(host, "repaired_vs_original", None)
    if metrics is not None:
        rows.append(_row("Max absolute error",
                         f"{metrics.max_absolute_error:.3e}"))
    # Only claim class agreement when the caller told us the model has it.
    agreement = getattr(host, "argmax_agreement", None)
    if agreement is not None:
        rows.append(_row("Argmax agreement", f"{100 * agreement:.2f}%"))

    reasons = list(getattr(host, "failure_reasons", []) or [])
    reasons += list(getattr(device, "failure_reasons", []) or [])
    note = ""
    if reasons:
        note = ('<div class="note warn">'
                + "<br>".join(esc(reason) for reason in reasons) + '</div>')

    return ('<section><h2>Correctness</h2><div class="card">'
            f'<table class="rows">{"".join(rows)}</table>{note}'
            '</div></section>')


def _benchmark(outcome) -> str:
    benchmark = outcome.benchmark
    if benchmark is None:
        return ""

    worst = max(benchmark.before.p50_ms, benchmark.after.p50_ms) or 1.0
    def bar(stats, tone):
        return (f'<div class="bar {tone}"><span '
                f'style="width:{100 * stats.p50_ms / worst:.1f}%"></span></div>')

    faster = benchmark.after.p50_ms < benchmark.before.p50_ms
    change = ((benchmark.after.p50_ms / benchmark.before.p50_ms - 1.0) * 100
              if benchmark.before.p50_ms else 0.0)
    change_text = (f"{abs(change):.1f}% {'lower' if change < 0 else 'higher'}")

    return (
        '<section><h2>Latency (p50)</h2><div class="card">'
        '<div style="margin-bottom:16px">'
        '<div class="metric-label">Before</div>'
        f'<div class="metric-value small">{_ms(benchmark.before.p50_ms)}</div>'
        f'{bar(benchmark.before, "")}'
        '</div>'
        '<div>'
        '<div class="metric-label">After</div>'
        f'<div class="metric-value small">{_ms(benchmark.after.p50_ms)}</div>'
        f'{bar(benchmark.after, "ok" if faster else "bad")}'
        '</div>'
        f'<table class="rows" style="margin-top:16px">'
        f'{_row("Change", change_text)}'
        f'{_row("p95", f"{benchmark.before.p95_ms:.2f} &rarr; {benchmark.after.p95_ms:.2f} ms")}'
        f'{_row("Samples", f"{benchmark.before.sample_count} per side")}'
        f'</table>'
        '</div></section>'
    )


def _pipeline(outcome) -> str:
    cells = []
    for stage in outcome.stages:
        tone = STATUS_TONE.get(stage.status, "idle")
        cells.append(
            f'<div class="stage">'
            f'<div class="stage-name">{esc(STAGE_SHORT.get(stage.name, stage.name))}</div>'
            f'<div class="stage-status {tone}">{esc(stage.status)}</div>'
            f'</div>'
        )
    return ('<section><h2>Pipeline</h2>'
            f'<div class="pipeline">{"".join(cells)}</div></section>')


def _details(outcome, executorch_version: str) -> str:
    parts = []

    parts.append("<h3>Pipeline stages</h3><table class='rows'>")
    for stage in outcome.stages:
        parts.append(_row(esc(stage.name),
                          esc(stage.status) + (f" &mdash; {esc(stage.detail)}"
                                               if stage.detail else ""),
                          wide=True))
    parts.append("</table>")

    if outcome.before_delegation is not None:
        before = outcome.before_delegation
        rows = [
            _row("Total operators", str(before.total_ops)),
            _row("Delegated", str(before.delegated_op_total)),
            _row("Portable", str(before.portable_op_total)),
            _row("Delegate blobs", str(before.delegate_blob_count)),
        ]
        if outcome.after_delegation is not None:
            after = outcome.after_delegation
            rows += [
                _row("Portable after repair", str(after.portable_op_total)),
                _row("Delegate blobs after repair", str(after.delegate_blob_count)),
            ]
        parts.append("<h3>Operator counts</h3><table class='rows'>"
                     + "".join(rows) + "</table>")

    profile = outcome.before_profile
    if profile is not None:
        parts.append(
            "<h3>Runtime breakdown</h3><table class='rows'>"
            + _row("Method::execute", _ms(profile.method_execute_ms))
            + _row("Inside delegate", _ms(profile.delegated_ms))
            + _row("Portable kernels", _ms(profile.portable_ms))
            + _row("Delegate calls", str(profile.delegate_call_count))
            + _row("Operator calls", str(profile.operator_call_count))
            + "</table>"
        )
        if profile.accounting_warning:
            parts.append(f'<div class="note warn">'
                         f'{esc(profile.accounting_warning)}</div>')
        if profile.portable_kernels:
            rows = "".join(
                _row(f'<span class="mono">{esc(kernel.operator_name)}</span>',
                     f"{kernel.total_ms:.2f} ms &middot; "
                     f"{_percent(kernel.runtime_fraction)} &middot; "
                     f"x{kernel.call_count}")
                for kernel in profile.portable_kernels
            )
            parts.append("<h3>All portable operators</h3>"
                         f"<table class='rows'>{rows}</table>")

    for rule_id, found in sorted(outcome.detections.items()):
        skipped = list(getattr(found, "skipped", []))
        if not skipped:
            continue
        rows = "".join(
            _row(f'<span class="mono">{esc(item.node_name)}</span>',
                 esc(item.reason), wide=True)
            for item in skipped
        )
        parts.append(f"<h3>{esc(rule_id)}: patterns declined</h3>"
                     f"<table class='rows'>{rows}</table>")

    if outcome.benchmark is not None:
        benchmark = outcome.benchmark
        parts.append(
            "<h3>Benchmark method</h3><table class='rows'>"
            + _row("Warmup iterations", f"{benchmark.warmup_iterations} per repetition")
            + _row("Measured iterations", f"{benchmark.measured_iterations} per repetition")
            + _row("Repetitions", f"{benchmark.repetitions}, interleaved before/after")
            + _row("Threads", str(benchmark.threads))
            + _row("Device", esc(benchmark.device_description), wide=True)
            + _row("Emulator", "YES" if benchmark.device_is_emulator else "NO")
            + "</table>"
        )

    artifacts = [_row("Run directory",
                      f'<span class="mono">{esc(outcome.run_dir)}</span>', wide=True)]
    if outcome.output_pte:
        artifacts.append(_row("Optimized program",
                              f'<span class="mono">{esc(outcome.output_pte)}</span>',
                              wide=True))
    artifacts.append(_row("ExecuTorch", esc(executorch_version)))
    parts.append("<h3>Artifacts</h3><table class='rows'>"
                 + "".join(artifacts) + "</table>")
    parts.append(
        '<div class="note">Raw ETDump traces, readable graphs, '
        '<span class="mono">results.json</span>, '
        '<span class="mono">verification.json</span> and '
        '<span class="mono">benchmark.json</span> are in the run directory.</div>'
    )

    return ("<details><summary>Technical details</summary>"
            f'<div class="details-body">{"".join(parts)}</div></details>')


# --- the page ---------------------------------------------------------------


def render(outcome, executorch_version: str = "") -> str:
    """Build the complete self-contained HTML document."""
    title = esc(outcome.model_name or "PyTorch Model")
    body = "".join([
        _masthead(outcome),
        _hero(outcome),
        _delegation(outcome),
        _hotspots(outcome),
        _repair_opportunity(outcome),
        _journey(outcome),
        _repair(outcome),
        _correctness(outcome),
        _benchmark(outcome),
        _target(outcome),
        _pipeline(outcome),
        _details(outcome, executorch_version),
        '<footer>Generated on this machine by DelegateDoctor. '
        'The Arm64 Android target executes, profiles and benchmarks; '
        'the report is never displayed there.</footer>',
    ])
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>DelegateDoctor - {title}</title>"
        f"<style>{CSS}</style>"
        f'</head><body><div class="page">{body}</div></body></html>\n'
    )


def generate_html_report(outcome, run_directory: str,
                         executorch_version: str = "") -> str:
    """Write `report.html` into the run directory and return its path."""
    os.makedirs(run_directory, exist_ok=True)
    path = os.path.join(run_directory, REPORT_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(render(outcome, executorch_version))
    return path
