"""Asking before anything private leaves the machine.

Two separate permissions, deliberately not interchangeable:

    SOURCE   sending model source, to prepare it for torch.export
    GRAPH    sending sanitized graph and profile metadata, to explore a repair

Agreeing to one is not agreeing to the other, and neither is implied by any
other flag in the tool. In particular `--yes`, which exists so Android setup
can install an SDK package without prompting, grants nothing here: consenting
to a download is not consenting to transmit your source code.

Each scope has its own flag, and neither covers the other:

    SOURCE   `--allow-ai-source`, and only in a run that cannot ask
    GRAPH    `--ai-repair`, which enables experimental AI repair at all

Every prompt defaults to **no**. A user who presses Enter without reading has
not agreed to send anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SCOPE_SOURCE = "source"
SCOPE_GRAPH = "graph"


class ConsentDeclined(RuntimeError):
    """The user did not agree to send this. Not an error in the tool."""


@dataclass
class ConsentDecision:
    granted: bool
    scope: str
    reason: str = ""


def _ask(prompt, question: str) -> bool:
    """A question whose default is no."""
    try:
        answer = prompt(question).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("y", "yes")


def source_disclosure(files, provider_name: str = "your configured AI provider") -> str:
    """Exactly what will be sent, named file by file."""
    listed = "\n".join(f"  {Path(path).name}" for path in files)
    return (
        f"DelegateDoctor AI Preparation\n"
        f"\n"
        f"To prepare this model for torch.export, DelegateDoctor needs to send\n"
        f"selected source code to {provider_name}.\n"
        f"\n"
        f"Files:\n"
        f"{listed}\n"
        f"\n"
        f"DelegateDoctor will NOT send:\n"
        f"  model weights\n"
        f"  checkpoint contents\n"
        f"  input tensors\n"
        f"  environment variables\n"
        f"  API keys\n"
        f"  unrelated project files\n"
    )


def graph_disclosure(opportunity=None) -> str:
    """The screen shown before asking to spend a provider request.

    Given a `RepairOpportunitySummary` this is the full decision screen: what
    the target measured, how much runs outside the delegate, what the worst
    operator costs, and the ceiling on removing it. That question - "is this
    worth an API call?" - cannot be answered from a privacy notice alone.

    A bare hotspot string still renders the disclosure, so callers that never
    profiled (and tests) keep working.
    """
    from ..repair_opportunity import RepairOpportunitySummary, format_decision_screen

    if isinstance(opportunity, RepairOpportunitySummary):
        return format_decision_screen(opportunity)

    subject = f"\n\nTop hotspot:\n  {opportunity}" if opportunity else ""
    return (
        f"DelegateDoctor AI Repair Exploration\n"
        f"\n"
        f"No DelegateDoctor catalog rule matches this hotspot.{subject}\n"
        f"\n"
        f"DelegateDoctor can send a sanitized neighbourhood of the exported\n"
        f"ATen graph and measured profiling metadata to your configured AI\n"
        f"provider to explore a candidate repair.\n"
        f"\n"
        f"It will NOT send:\n"
        f"  model weights\n"
        f"  tensor values\n"
        f"  representative inputs\n"
        f"  checkpoints\n"
        f"  your model source\n"
        f"  API keys\n"
    )


def request_source_consent(files, interactive: bool, preapproved: bool,
                           announce=print, prompt=input) -> ConsentDecision:
    """May DelegateDoctor send these source files?

    `preapproved` is the explicit non-interactive opt-in
    (`--allow-ai-source`).
    Nothing else grants it: not `--yes`, not a previous run, not consent given
    for a different scope.
    """
    if preapproved:
        announce(source_disclosure(files))
        announce("Sending source (--allow-ai-source).\n")
        return ConsentDecision(True, SCOPE_SOURCE, "explicit flag")

    if not interactive:
        return ConsentDecision(False, SCOPE_SOURCE, (
            "AI preparation needs to send model source to your AI provider, and\n"
            "this run is non-interactive.\n"
            "\n"
            "Re-run with --non-interactive --allow-ai-source to permit it, or\n"
            "skip AI entirely by using the Python API, which never contacts a\n"
            "provider:\n"
            "\n"
            "    from delegate_doctor import optimize\n"
            "    result = optimize(model, args=(example_input,))"
        ))

    announce(source_disclosure(files))
    if _ask(prompt, "Continue? [y/N]: "):
        return ConsentDecision(True, SCOPE_SOURCE, "interactive")
    return ConsentDecision(False, SCOPE_SOURCE, (
        "Source was not sent.\n"
        "\n"
        "To analyze this model without AI, call the Python API on the live\n"
        "module - that path never contacts a provider:\n"
        "\n"
        "    from delegate_doctor import optimize\n"
        "    result = optimize(model, args=(example_input,))"
    ))


def request_additional_file_consent(file_name: str, interactive: bool,
                                    preapproved: bool, announce=print,
                                    prompt=input) -> ConsentDecision:
    """One extra local file, asked for by name and one at a time.

    Consent for `model.py` never silently extends to its imports: each file is
    a separate question, so "send this project" can never be answered by
    accident.
    """
    if preapproved:
        announce(f"Also sending {file_name} (--allow-ai-source).")
        return ConsentDecision(True, SCOPE_SOURCE, "explicit flag")

    if not interactive:
        return ConsentDecision(False, SCOPE_SOURCE,
                               f"{file_name} was needed but could not be requested "
                               f"in a non-interactive run")

    announce(
        f"\nDelegateDoctor needs one additional source file to understand the "
        f"model:\n"
        f"\n"
        f"  {file_name}\n"
    )
    if _ask(prompt, "Send this file to the configured AI provider? [y/N]: "):
        return ConsentDecision(True, SCOPE_SOURCE, "interactive")
    return ConsentDecision(False, SCOPE_SOURCE, f"{file_name} was not sent")


def repair_privacy_notice(configuration=None) -> str:
    """What an experimental repair request contains. Stated, not asked.

    `--ai-repair` is the authorization: the user typed it. Prompting again
    would be a second confirmation of one decision, and would make the flag
    unusable in a non-interactive run - which is exactly where an explicit
    opt-in flag is most useful.

    What does *not* go away is the obligation to say what leaves the machine.
    That is a disclosure, and it is printed once, before the first request.
    """
    destination = "your configured AI provider"
    if configuration is not None:
        if getattr(configuration, "is_local", False):
            destination = "your local provider"
        else:
            destination = configuration.describe()

    return (
        f"\nDelegateDoctor may send to {destination}:\n"
        f"  operator names\n"
        f"  graph relationships\n"
        f"  tensor shapes and dtypes\n"
        f"  profiling metadata\n"
        f"\n"
        f"DelegateDoctor will NOT send:\n"
        f"  model source\n"
        f"  weights\n"
        f"  tensor values\n"
        f"  representative inputs\n"
        f"  checkpoints\n"
        f"  API keys\n"
    )


def request_repair_consent(opportunity, interactive: bool, preapproved: bool,
                           announce=print, prompt=input) -> ConsentDecision:
    """May DelegateDoctor send graph metadata to explore a repair?

    Separate from source consent on purpose. A user who agreed to have their
    model prepared has not thereby agreed to an exploratory repair round.

    The measurements are shown even when `--ai-repair` already granted
    permission: a user who pre-approved exploration should still see what the
    request is being spent on, and whether the ceiling justified it.
    """
    if preapproved:
        announce(graph_disclosure(opportunity))
        announce("Exploring an AI repair (--ai-repair).\n")
        return ConsentDecision(True, SCOPE_GRAPH, "explicit flag")

    if not interactive:
        return ConsentDecision(False, SCOPE_GRAPH,
                               "AI repair exploration was not enabled "
                               "for this run")

    announce(graph_disclosure(opportunity))
    if _ask(prompt, "Explore an AI repair? [y/N]: "):
        return ConsentDecision(True, SCOPE_GRAPH, "interactive")
    return ConsentDecision(False, SCOPE_GRAPH, "AI repair exploration declined")
