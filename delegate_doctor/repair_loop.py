"""Optimization as a sequence, not a single repair.

One repair is rarely the whole story. Removing a 38% softmax hotspot changes
every other percentage in the profile, can expose a fallback that was hidden
behind it, and can change the graph enough that the *next* worst operator is
one that did not appear in the original ranking at all.

So DelegateDoctor iterates:

    profile the current accepted program
      -> rank portable hotspots by measured runtime share
      -> group them by the catalog rule that recognises them
      -> attempt the costliest rule once, rewriting every site it matches
      -> if no rule matches and --ai-repair was passed, explore with AI
      -> verify, benchmark, keep or reject
      -> if kept: re-profile, and rank again from the new measurement

The scheduling unit is the *rule*, not the site. A rule's `apply()` has always
been graph-wide, so asking it once per matching hotspot produced the identical
candidate over and over - nine report rows and nine device benchmarks for one
decision. `CatalogRuleMatch` below is what fixed that.

The re-profile is the point. A list built from the original graph is stale the
moment a repair is accepted, and applying several repairs from one stale
ranking would be optimizing a model that no longer exists.

Three programs, kept distinct
-----------------------------

    ORIGINAL           what preparation produced. Never mutated. The semantic
                       reference every candidate is ultimately judged against.
    CURRENT ACCEPTED   original plus every repair that passed its gates.
    CANDIDATE          a deep copy of CURRENT with one proposed repair applied.

A candidate becomes the new CURRENT only after passing every existing gate.
Correctness is always measured against ORIGINAL - never against the previous
accepted step - so three repairs each drifting a little cannot add up to a
model that is wrong while every individual step looked fine.

Performance is the opposite: each candidate is benchmarked against the CURRENT
accepted program, because the question is whether *this* repair earns its
place on top of what is already there. The original-to-final comparison is
reported separately, as the cumulative result.

This module owns the policy - eligibility, ranking, identity, termination,
bookkeeping - and none of the mechanics. Lowering, profiling, verification and
benchmarking are injected, so every rule below is unit-testable with no device.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

from . import operator_correlation

# Two thresholds, because the two kinds of repair cost different things.
#
# A known DelegateDoctor repair is deterministic, free, and already understood:
# applying one costs a lowering, a verification and a benchmark, all of which
# DelegateDoctor was going to run anyway. So the bar is low - a tenth of a
# percent - and the benchmark gate throws out anything that did not help.
MIN_DD_HOTSPOT_RUNTIME_SHARE = 0.001

# There is deliberately NO per-site or per-operator AI threshold.
#
# AI eligibility is a question about the whole model - see
# `model_exploration.MIN_AI_PORTABLE_RUNTIME_SHARE` - because deciding which
# combination of operators is worth repairing is the investigation, not a
# precondition for starting it. Three operators at 3%, 2% and 2% are a 7%
# opportunity that no per-operator bar would ever admit.
#
# The floor above is the only per-site filter: it keeps measurement noise out
# of the hotspot ranking that catalog matching and the report both read.

# A safety limit, not a product limitation. Every normal run terminates long
# before this because hotspots are marked attempted and never revisited; this
# catches a pathological case where accepted repairs keep producing new ones.
MAX_REPAIR_ITERATIONS = 16

# --- what became of one hotspot ---------------------------------------------

ACCEPTED = "ACCEPTED"
REJECTED = "REJECTED"
SKIPPED_BY_USER = "AI_SKIPPED_BY_USER"
# Reached one of DelegateDoctor's own checks and did not justify the device
# work. Distinct from REJECTED, which means the gates actually ran.
SKIPPED = "SKIPPED"
AI_UNAVAILABLE = "AI_UNAVAILABLE"
NO_CANDIDATE = "NO_CANDIDATE"
NOT_APPLICABLE = "NOT_APPLICABLE"

# Where a repair came from.
SOURCE_CATALOG = "catalog"
SOURCE_AI = "ai"
SOURCE_NONE = "none"

# Outcomes after which a hotspot is finished for this run. A hotspot that was
# rejected, declined or had no candidate must not be tried again: re-offering
# it would prompt the same question forever and re-spend the same API request.
CONCLUSIVE = frozenset({REJECTED, SKIPPED, SKIPPED_BY_USER, AI_UNAVAILABLE,
                        NO_CANDIDATE, NOT_APPLICABLE})


@dataclass(frozen=True)
class RepairHotspot:
    """One measured portable hotspot, identified well enough to track.

    An operator name alone is not an identity: the same operator can appear at
    several places in a graph, with very different costs and very different
    repairs. The node name from the exported graph pins it down, and the pair
    survives repairs applied elsewhere in the graph.
    """

    operator_name: str
    kernel_name: str
    runtime_share: float
    event_time_ms: float
    node_id: str = ""
    catalog_match: Optional[str] = None

    # How the runtime operator was mapped back to the graph, and what it was
    # mapped to. Carried so a hotspot that could not be targeted can say why,
    # rather than being reported as an operator that does not exist.
    resolution: object = None

    @property
    def hotspot_id(self) -> str:
        """Stable across iterations, so an attempt can be remembered."""
        return f"{self.node_id or 'unlocated'}:{self.operator_name}"

    @property
    def targetable(self) -> bool:
        """Did this hotspot resolve to exactly one graph node?

        A hotspot that did not is still a real finding - it was measured, and
        it is reported - but nothing may be rewritten on the strength of it.
        """
        return bool(self.node_id)

    @property
    def resolution_status(self) -> str:
        if self.resolution is None:
            return operator_correlation.UNRESOLVED
        return self.resolution.status

    @property
    def resolution_reason(self) -> str:
        return getattr(self.resolution, "reason", "") or ""

    @property
    def eligible_for_catalog(self) -> bool:
        """Worth a deterministic repair. Exactly 0.5% does not qualify."""
        return self.runtime_share > MIN_DD_HOTSPOT_RUNTIME_SHARE

    @property
    def eligible(self) -> bool:
        """Worth collecting at all.

        One floor for both routes. A site that clears it is a real measurement:
        catalog matching looks at it, and the report ranks it. Whether AI is
        asked is a separate question about the model as a whole.
        """
        return self.eligible_for_catalog

    @property
    def theoretical_upper_bound(self) -> Optional[float]:
        """Amdahl's ceiling if this hotspot's cost became zero.

        A bound, never an expected or predicted speedup: it assumes the
        operator becomes free and nothing else changes.
        """
        share = self.runtime_share
        if share is None or share <= 0.0 or share >= 1.0:
            return None
        return 1.0 / (1.0 - share)

    def describe(self) -> str:
        return (f"{self.operator_name}  {100 * self.runtime_share:.1f}%  "
                f"{self.event_time_ms:.3f} ms")

    def to_dict(self) -> dict:
        return {
            "hotspot_id": self.hotspot_id,
            "node_id": self.node_id,
            "operator": self.operator_name,
            "runtime_share": self.runtime_share,
            "event_time_ms": self.event_time_ms,
            "catalog_match": self.catalog_match,
            "theoretical_upper_bound": self.theoretical_upper_bound,
            "resolution": (self.resolution.to_dict()
                           if self.resolution is not None else None),
        }


@dataclass
class RepairAttempt:
    """One proposal for one hotspot, and every gate it met.

    Recorded whatever the outcome. A rejected repair is evidence - it says the
    pattern was recognised and the target disagreed - and hiding it would make
    the report a highlight reel.
    """

    iteration: int
    # None for a model-level AI candidate: an AI proposal is about the graph,
    # not about one profiled hotspot, and pretending otherwise put a
    # meaningless operator name on every AI row.
    hotspot: Optional[RepairHotspot]
    source: str = SOURCE_NONE
    repair_id: str = ""
    candidate_id: str = ""
    status: str = NOT_APPLICABLE
    reason: str = ""

    # A catalog rule is applied to every site it recognises, so an attempt is
    # about a set of sites rather than one. `matching_sites` is what `apply()`
    # actually rewrote; `measured_sites` is how many profiled hotspots this
    # attempt speaks for; `represented_runtime` is their aggregate share. They
    # differ legitimately - a rule may rewrite a site too cheap to profile.
    matching_sites: Optional[int] = None
    measured_sites: Optional[int] = None
    represented_runtime: Optional[float] = None

    host_verification_passed: Optional[bool] = None
    device_verification_passed: Optional[bool] = None
    # How well the backend reproduced the host. A property of the model and the
    # backend, not of this repair - kept separate so it can never be read as a
    # repair failure. See device_verification.classify_backend_fidelity.
    backend_fidelity: str = ""
    backend_fidelity_reason: str = ""
    before_latency_ms: Optional[float] = None
    after_latency_ms: Optional[float] = None

    @property
    def accepted(self) -> bool:
        return self.status == ACCEPTED

    @property
    def speedup(self) -> Optional[float]:
        if not self.before_latency_ms or not self.after_latency_ms:
            return None
        return self.before_latency_ms / self.after_latency_ms

    @property
    def label(self) -> str:
        return self.repair_id or self.candidate_id or "no repair"

    @property
    def subject(self) -> str:
        """What this attempt was about, for a report row.

        A catalog repair is about a measured hotspot; an AI candidate is about
        the model. Naming an operator on an AI row would imply a targeting
        decision DelegateDoctor did not make.
        """
        if self.hotspot is not None:
            return self.hotspot.operator_name
        return "model-level exploration"

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "hotspot": (self.hotspot.to_dict()
                        if self.hotspot is not None else None),
            "subject": self.subject,
            "source": self.source,
            "repair_id": self.repair_id,
            "candidate_id": self.candidate_id,
            "status": self.status,
            "reason": self.reason,
            "matching_sites": self.matching_sites,
            "measured_sites": self.measured_sites,
            "represented_runtime": self.represented_runtime,
            "host_verification": self.host_verification_passed,
            "device_verification": self.device_verification_passed,
            "backend_fidelity": self.backend_fidelity or None,
            "backend_fidelity_reason": self.backend_fidelity_reason or None,
            "before_latency_ms": self.before_latency_ms,
            "after_latency_ms": self.after_latency_ms,
            "speedup": self.speedup,
        }


# How AI figured in one run. Asked at most once, whatever the hotspot count.
AI_CONSENT_NOT_NEEDED = "not needed"
# Experimental AI repair was not opted into. The default product, and not a
# failure of anything.
AI_CONSENT_NOT_ENABLED = "not enabled"
AI_CONSENT_GRANTED = "granted"
AI_CONSENT_DECLINED = "declined"
AI_CONSENT_UNAVAILABLE = "unavailable"


@dataclass
class RepairHistory:
    """Every hotspot considered in one run, in the order it was considered."""

    attempts: list = field(default_factory=list)
    iterations: int = 0
    stop_reason: str = ""

    # One decision covering the whole run. A user who agreed to AI exploration
    # agreed to it for this model, not for one operator - re-asking after every
    # re-profile would be nagging, not consent.
    ai_consent: str = AI_CONSENT_NOT_NEEDED
    ai_hotspots_offered: int = 0
    ai_families_offered: int = 0

    # What the provider call resolved to, and how many proposals it actually
    # produced. Kept apart from the repair attempts: a provider that never
    # answered proposed nothing, and counting it as a candidate was the bug
    # that made "empty response" read as "one candidate, none accepted".
    ai_provider_status: str = ""
    ai_provider_detail: str = ""
    ai_candidates_proposed: int = 0
    ai_candidates_tested: int = 0

    # Measured on the original program, before anything was applied.
    original_latency_ms: Optional[float] = None
    original_operator_delegation: Optional[float] = None
    original_runtime_delegation: Optional[float] = None

    # Measured on the final accepted program.
    final_latency_ms: Optional[float] = None
    final_operator_delegation: Optional[float] = None
    final_runtime_delegation: Optional[float] = None

    @property
    def accepted(self) -> list:
        return [attempt for attempt in self.attempts if attempt.accepted]

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def catalog_count(self) -> int:
        return sum(1 for attempt in self.accepted
                   if attempt.source == SOURCE_CATALOG)

    @property
    def ai_count(self) -> int:
        return sum(1 for attempt in self.accepted if attempt.source == SOURCE_AI)

    @property
    def rejected_count(self) -> int:
        return sum(1 for attempt in self.attempts if attempt.status == REJECTED)

    @property
    def any_accepted(self) -> bool:
        return bool(self.accepted)

    @property
    def total_speedup(self) -> Optional[float]:
        """Original to final. The number a user actually ships against."""
        if not self.original_latency_ms or not self.final_latency_ms:
            return None
        return self.original_latency_ms / self.final_latency_ms

    @property
    def applied_repair_ids(self) -> list:
        return [attempt.label for attempt in self.accepted]

    def record(self, attempt: RepairAttempt) -> RepairAttempt:
        self.attempts.append(attempt)
        return attempt

    def to_dict(self) -> dict:
        return {
            "iterations": self.iterations,
            "stop_reason": self.stop_reason,
            "ai_consent": self.ai_consent,
            "ai_hotspots_offered": self.ai_hotspots_offered,
            "ai_families_offered": self.ai_families_offered,
            "ai_provider_status": self.ai_provider_status,
            "ai_provider_detail": self.ai_provider_detail,
            "ai_candidates_proposed": self.ai_candidates_proposed,
            "ai_candidates_tested": self.ai_candidates_tested,
            "accepted": self.accepted_count,
            "catalog_repairs": self.catalog_count,
            "ai_repairs": self.ai_count,
            "rejected": self.rejected_count,
            "original_latency_ms": self.original_latency_ms,
            "final_latency_ms": self.final_latency_ms,
            "total_speedup": self.total_speedup,
            "original_operator_delegation": self.original_operator_delegation,
            "final_operator_delegation": self.final_operator_delegation,
            "original_runtime_delegation": self.original_runtime_delegation,
            "final_runtime_delegation": self.final_runtime_delegation,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


# --- building the hotspot list ------------------------------------------------


def locate_node(exported_program, operator_name: str,
                skip: frozenset = frozenset()) -> str:
    """The graph node this profiled operator refers to, or "" if not unique.

    Thin wrapper over `operator_correlation.resolve_hotspot`, kept because
    "give me the node id" is what most callers want. Anything that needs to
    distinguish "no such node" from "seventeen of them" should call the
    resolver directly and read the structured result.
    """
    resolution = operator_correlation.resolve_hotspot(
        exported_program, operator_name, exclude=skip)
    return resolution.node_id


def collect_hotspots(profile, exported_program=None,
                     catalog_lookup=None) -> list:
    """Eligible portable hotspots for the current program, worst first.

    Only *portable* kernels are considered. Delegated runtime is work that
    already went through XNNPACK, and an expensive delegated operator is not a
    repair opportunity - it is the backend doing its job.

    An operator measured at several graph sites becomes several hotspots, each
    carrying its own site's cost and its own node. The aggregate stays in the
    profile for reporting; targeting works on sites, because a repair is
    applied to a node and "layer norm is 10.1% of runtime" does not say which
    of twenty-four layer norms to rewrite.
    """
    kernels = list(getattr(profile, "portable_kernels", ()) or ())
    kernels.sort(key=lambda kernel: kernel.runtime_fraction, reverse=True)

    if exported_program is None:
        # No graph to correlate against: report the operators as measured,
        # unresolved. Nothing will be rewritten on that basis.
        return [hotspot for hotspot in
                (_unlocated_hotspot(kernel, catalog_lookup)
                 for kernel in kernels) if hotspot.eligible]

    claimed = set()
    hotspots = []
    for kernel in kernels:
        hotspots += _hotspots_for(kernel, exported_program, catalog_lookup,
                                  claimed)

    # Site costs, not operator totals, decide the order: the highest-cost
    # resolvable site is the one worth repairing first.
    hotspots.sort(key=lambda hotspot: hotspot.runtime_share, reverse=True)
    return [hotspot for hotspot in hotspots if hotspot.eligible]


def _unlocated_hotspot(kernel, catalog_lookup) -> RepairHotspot:
    return RepairHotspot(
        operator_name=kernel.operator_name,
        kernel_name=kernel.name,
        runtime_share=kernel.runtime_fraction,
        event_time_ms=kernel.total_ms,
        catalog_match=(catalog_lookup(kernel.name) if catalog_lookup else None),
    )


def _hotspots_for(kernel, exported_program, catalog_lookup,
                  claimed: set) -> list:
    """One kernel, correlated to the graph. May produce several hotspots."""
    catalog_match = catalog_lookup(kernel.name) if catalog_lookup else None
    site_costs = tuple(getattr(kernel, "site_costs", ()) or ())
    site_count = getattr(kernel, "site_count", None)
    debug_node_id = str(getattr(kernel, "debug_node_id", "") or "")

    candidates = operator_correlation.candidate_nodes(
        exported_program, kernel.operator_name)
    unclaimed = [str(node.name) for node in candidates
                 if str(node.name) not in claimed]

    # One site, or no per-site breakdown: a single hotspot, resolved however
    # the resolver can manage.
    if len(site_costs) <= 1 or len(unclaimed) != len(site_costs):
        resolution = operator_correlation.resolve_hotspot(
            exported_program, kernel.operator_name,
            debug_node_id=debug_node_id, site_count=site_count,
            exclude=frozenset(claimed))
        if resolution.resolved:
            claimed.add(resolution.node_id)
        return [RepairHotspot(
            operator_name=kernel.operator_name,
            kernel_name=kernel.name,
            runtime_share=kernel.runtime_fraction,
            event_time_ms=kernel.total_ms,
            node_id=resolution.node_id,
            catalog_match=catalog_match,
            resolution=resolution,
        )]

    # As many measured sites as unclaimed candidate nodes: the trace and the
    # graph agree, so each site is the correspondingly-ordered node. Each gets
    # its own share of the operator's measured runtime.
    total = kernel.total_ms or 0.0
    hotspots = []
    for index, (node_id, cost) in enumerate(zip(unclaimed, site_costs)):
        claimed.add(node_id)
        share = (kernel.runtime_fraction * (cost / total)) if total else 0.0
        hotspots.append(RepairHotspot(
            operator_name=kernel.operator_name,
            kernel_name=kernel.name,
            runtime_share=share,
            event_time_ms=cost,
            node_id=node_id,
            catalog_match=catalog_match,
            resolution=operator_correlation.HotspotResolution(
                runtime_operator=kernel.operator_name,
                canonical=operator_correlation.canonical_operator(
                    kernel.operator_name),
                status=operator_correlation.RESOLVED,
                node_ids=(node_id,),
                detail=(f"site {index + 1} of {len(site_costs)} matched to "
                        f"graph node {index + 1} in execution order"),
            ),
        ))
    return hotspots


def catalog_lookup_for(rules) -> callable:
    """A `kernel_name -> rule id` function built from the rule catalog.

    Kept as a closure so the loop never imports the rules and never needs to
    know what a rule is beyond "something with an id that recognises a kernel".
    """
    def lookup(kernel_name: str):
        for rule in rules:
            try:
                if rule.matches_portable_kernel(kernel_name):
                    return rule.RULE_ID
            except Exception:
                continue
        return None

    return lookup


def next_catalog_hotspot(hotspots: list, finished: set) -> Optional[RepairHotspot]:
    """The costliest hotspot a catalog rule recognises, if any is left.

    Kept for callers that want a single site. The loop itself schedules by
    *rule* - see `next_catalog_match` - because a rule is not a site.
    """
    for hotspot in hotspots:
        if (hotspot.catalog_match and hotspot.eligible_for_catalog
                and hotspot.hotspot_id not in finished):
            return hotspot
    return None


# --- the catalog scheduling unit ----------------------------------------------


@dataclass(frozen=True)
class CatalogRuleMatch:
    """One catalog rule, and every measured hotspot on this graph it claims.

    This is the unit of catalog repair, and the fix for a real bug: the loop
    used to schedule by hotspot, so a rule matching nine sites was attempted
    nine times. Every attempt called the same deterministic `apply()`, which
    rewrites every site it recognises, so all nine produced an identical
    candidate, met identical gates and were rejected identically - nine rows in
    the report for one decision.

    A rule is not a site. `apply()` has always been graph-wide; the scheduler is
    what disagreed.
    """

    rule_id: str
    hotspots: tuple

    @property
    def primary(self) -> RepairHotspot:
        """The costliest matched site, which names the attempt."""
        return self.hotspots[0]

    @property
    def operator_name(self) -> str:
        return self.primary.operator_name

    @property
    def measured_site_count(self) -> int:
        return len(self.hotspots)

    @property
    def runtime_share(self) -> float:
        """Aggregate measured runtime this rule is responsible for.

        Safe to sum: `collect_hotspots` claims each graph node at most once, so
        two hotspots never describe the same measured work. A kernel that could
        not be split into sites contributes one hotspot carrying its whole
        fraction; a kernel that could contributes shares that add back up to it.
        Either way each measured event is counted once.

        Clamped to 1.0 regardless, because a runtime share above 100% would be
        a reporting claim no measurement can support.
        """
        return min(1.0, sum(hotspot.runtime_share for hotspot in self.hotspots))

    @property
    def event_time_ms(self) -> float:
        return sum(hotspot.event_time_ms for hotspot in self.hotspots)

    @property
    def hotspot_ids(self) -> tuple:
        return tuple(hotspot.hotspot_id for hotspot in self.hotspots)

    def describe(self) -> str:
        return (f"{self.rule_id} on {self.operator_name}: "
                f"{self.measured_site_count} measured site"
                f"{'' if self.measured_site_count == 1 else 's'}, "
                f"{100 * self.runtime_share:.1f}% of runtime")


def group_catalog_matches(hotspots: list) -> list:
    """Collapse eligible hotspots into one match per catalog rule, worst first.

    Ordering is by aggregate share, not by any single site's: a rule matching
    nine sites worth 60% collectively outranks one site worth 20%, which is the
    order a user would choose if asked.
    """
    grouped: dict = {}
    for hotspot in hotspots:
        if not hotspot.catalog_match or not hotspot.eligible_for_catalog:
            continue
        grouped.setdefault(hotspot.catalog_match, []).append(hotspot)

    matches = [CatalogRuleMatch(rule_id=rule_id, hotspots=tuple(sites))
               for rule_id, sites in grouped.items()]
    matches.sort(key=lambda match: match.runtime_share, reverse=True)
    return matches


def next_catalog_match(hotspots: list, attempted=frozenset(),
                       fingerprint: str = "") -> Optional[CatalogRuleMatch]:
    """The costliest catalog rule not yet attempted on this graph state.

    Searched before AI at *every* fresh profile, including the one taken after
    an AI repair was accepted. A deterministic repair that is already understood
    should never wait behind a request to a provider.

    `attempted` holds `attempt_key(fingerprint, rule_id)` values, so a rule that
    was rejected is finished for *this* graph while remaining eligible on a
    different one - which is what makes an accepted repair able to expose more
    work for the same rule without letting a rejected one retry forever.
    """
    for match in group_catalog_matches(hotspots):
        if attempt_key(fingerprint, match.rule_id) not in attempted:
            return match
    return None


def graph_fingerprint(exported_program) -> str:
    """Identity of a graph state, for deciding whether a rule has been tried.

    The printed graph is the natural fingerprint: it changes exactly when the
    nodes, their arguments or their connections change, which is exactly when a
    rule that produced nothing before might produce something now.

    Returns "" when the graph cannot be rendered. That is not a failure - the
    dedup key degrades to the rule id alone, which is the conservative
    direction: a rule is attempted once rather than repeatedly.
    """
    try:
        printed = str(exported_program.graph)
    except Exception:
        return ""
    return hashlib.sha256(printed.encode("utf-8", "replace")).hexdigest()[:16]


def attempt_key(fingerprint: str, rule_id: str) -> str:
    """The identity of "this rule, on this graph state"."""
    return f"{fingerprint}:{rule_id}"


def eligible_ai_hotspots(hotspots: list, finished: set) -> list:
    """Every unattempted site no catalog rule recognises, worst first.

    No per-site runtime filter: this is the diagnostic list of fallback the
    catalog did not recognise, which the report shows and which tells the
    model-level AI question that there is something left to investigate.

    A site with a catalog match never appears here - not even one whose catalog
    repair was rejected. The rule is the answer for that operator, and a
    rejection is an answer too, not an invitation to guess.
    """
    return [hotspot for hotspot in hotspots
            if not hotspot.catalog_match
            and hotspot.hotspot_id not in finished]


def next_hotspot(hotspots: list, finished: set) -> Optional[RepairHotspot]:
    """The highest-priority hotspot not already conclusively attempted.

    Catalog first, then AI - the same order the loop itself uses.
    """
    catalog = next_catalog_hotspot(hotspots, finished)
    if catalog is not None:
        return catalog
    remaining = eligible_ai_hotspots(hotspots, finished)
    return remaining[0] if remaining else None


# --- candidate identity --------------------------------------------------------


class CandidateNumbering:
    """Globally unique AI candidate IDs for one optimization run.

    Numbering does not restart per hotspot: `AI-CANDIDATE-003` should mean one
    thing in a report, not "the third candidate for whichever hotspot this line
    is under".
    """

    def __init__(self, prefix: str = "AI-CANDIDATE"):
        self.prefix = prefix
        self._issued = 0

    def next(self) -> str:
        self._issued += 1
        return f"{self.prefix}-{self._issued:03d}"

    @property
    def issued(self) -> int:
        return self._issued


# --- terminal rendering --------------------------------------------------------


def _header_line(index: int, operator_name: str, runtime_share: float) -> str:
    return (f"\n[{index}] {operator_name}"
            f"{' ' * max(1, 24 - len(operator_name))}"
            f"{100 * runtime_share:.1f}%")


def format_hotspot_header(index: int, hotspot: RepairHotspot) -> str:
    """The one-line opener for a repair opportunity."""
    return _header_line(index, hotspot.operator_name, hotspot.runtime_share)


def format_match_header(index: int, match: CatalogRuleMatch,
                        verbose: bool = False) -> str:
    """The opener for a catalog rule attempt: the operator and its total cost.

    One line whatever the site count. Nine sites at 10.9%, 9.7%, ... are one
    60.9% opportunity, and printing them as nine headers described the loop's
    scheduling rather than the model's behaviour.

    `--verbose` adds the per-site breakdown, which is genuinely useful when the
    aggregate looks surprising, and noise otherwise.
    """
    text = _header_line(index, match.operator_name, match.runtime_share)
    if verbose and match.measured_site_count > 1:
        for position, hotspot in enumerate(match.hotspots, start=1):
            text += (f"\n      site {position}: "
                     f"{hotspot.node_id or 'unlocated'}  "
                     f"{100 * hotspot.runtime_share:.1f}%  "
                     f"{hotspot.event_time_ms:.3f} ms")
    return text


def format_attempt_result(attempt: RepairAttempt) -> str:
    """What happened, in the compact form the terminal uses.

    One column for everything a repair attempt prints, so the header and the
    gates read as a single block.
    """
    lines = []
    if attempt.host_verification_passed is not None:
        lines.append(f"    Host correctness       "
                     f"{'PASS' if attempt.host_verification_passed else 'FAIL'}")
    if attempt.device_verification_passed is not None:
        lines.append(f"    Device correctness     "
                     f"{'PASS' if attempt.device_verification_passed else 'FAIL'}")
    # Only when there is something to say. A backend that reproduces its host
    # result is the expected case and does not need a line of its own.
    if attempt.backend_fidelity and attempt.backend_fidelity != "OK":
        lines.append(f"    Backend fidelity       {attempt.backend_fidelity}")
        if attempt.backend_fidelity_reason:
            lines.append(f"      {attempt.backend_fidelity_reason}")
    if attempt.before_latency_ms and attempt.after_latency_ms:
        lines.append(f"    p50                    "
                     f"{attempt.before_latency_ms:.2f} -> "
                     f"{attempt.after_latency_ms:.2f} ms")
    lines.append(f"    Result                 {attempt.status}")
    if attempt.reason:
        lines.append(f"    Reason                 {attempt.reason}")
    return "\n".join(lines)


def _ai_sequence_note(history: RepairHistory) -> str:
    """One line about experimental AI repair, only when it is worth saying."""
    if history.ai_consent == AI_CONSENT_NOT_ENABLED:
        return "Experimental AI repair was not enabled."
    if history.ai_provider_status == "NO_REPAIR_PROPOSED":
        return "AI model exploration completed with no repair proposal."
    if history.ai_provider_status:
        return f"AI model exploration: {history.ai_provider_status}."
    return ""


def format_history(history: RepairHistory) -> str:
    """The optimization sequence, for `report.txt`."""
    lines = ["", "OPTIMIZATION SEQUENCE", "-" * 40]

    if not history.attempts:
        # What happened, not a threshold claim. The previous text asserted
        # portable runtime was below the AI threshold, and printed exactly
        # that on a run measured at 60.8% portable - because it was a fixed
        # string rather than a reading of the run.
        lines.append("No known DelegateDoctor repair matched.")
        note = _ai_sequence_note(history)
        if note:
            lines.append(note)
        return "\n".join(lines)

    for position, attempt in enumerate(history.attempts, start=1):
        lines.append(
            f"{position}. {attempt.label}  ({attempt.source})")
        lines.append(f"   subject              {attempt.subject}")
        if attempt.matching_sites is not None:
            lines.append(f"   matching sites       {attempt.matching_sites}")
        if attempt.represented_runtime is not None:
            lines.append(f"   represented runtime  "
                         f"{100 * attempt.represented_runtime:.1f}%")
        elif attempt.hotspot is not None:
            lines.append(f"   runtime share before "
                         f"{100 * attempt.hotspot.runtime_share:.1f}%")
        if attempt.backend_fidelity and attempt.backend_fidelity != "OK":
            lines.append(f"   backend fidelity     {attempt.backend_fidelity}")
        if attempt.before_latency_ms and attempt.after_latency_ms:
            lines.append(f"   p50                  "
                         f"{attempt.before_latency_ms:.2f} -> "
                         f"{attempt.after_latency_ms:.2f} ms")
        lines.append(f"   result               {attempt.status}")
        if attempt.reason:
            lines.append(f"   reason               {attempt.reason}")

    lines += ["", f"Iterations              {history.iterations}",
              f"Accepted repairs        {history.accepted_count}",
              f"  catalog               {history.catalog_count}",
              f"  AI                    {history.ai_count}",
              f"Rejected                {history.rejected_count}",
              f"AI consent              {history.ai_consent}"]
    if history.stop_reason:
        lines.append(f"Stopped because         {history.stop_reason}")

    if history.original_latency_ms and history.final_latency_ms:
        lines += ["",
                  f"Original p50            {history.original_latency_ms:.2f} ms",
                  f"Final p50               {history.final_latency_ms:.2f} ms"]
        if history.total_speedup:
            lines.append(f"Total speedup           "
                         f"{history.total_speedup:.2f}x")
    return "\n".join(lines)


