"""The `delegate-doctor` command line interface.

One command does the work:

    delegate-doctor optimize model.py

There is no AI workflow and no separate no-AI workflow. There is one workflow,
and it tries the deterministic thing first every time:

    model.py
      -> does it declare the DelegateDoctor model interface?
           yes -> run it in a child process, torch.export, done. No provider is
                  constructed and no credential is read.
           no  -> AI preparation is offered, if a provider is configured
      -> ExecuTorch/XNNPACK, Arm profiling
      -> known repairs first, applied automatically: one rule attempt per
         graph state, for any rule above 0.1% of measured runtime
      -> then, only with --ai-repair, one bounded AI exploration of the
         whole model, once per run
      -> verify against the original, benchmark against the current best
      -> keep only what is correct and faster, then re-profile and start again
         from the known repairs

AI is a capability, not a mode. It is reached only where a deterministic answer
does not exist: a model source that does not say how to build itself, or a
measured hotspot no catalog rule recognises. A run with no provider configured
is a complete run, not a degraded one.

    The agent proposes. DelegateDoctor verifies. The Arm target decides.

`setup-android` provisions the Arm64 measurement environment, and `optimize`
can trigger it on demand, so reading about it first is optional.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

from . import (android_setup, device, environment_check, model_interface,
               model_source, pipeline, target_selection)
from .agent import consent, credentials
from .agent.client import AIError
from .agent.preparation import PreparationError
from .api import ExportFailed
from .model_source import ModelSourceError
from .result import exit_code

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)
DEFAULT_RUNNERS_DIR = pipeline.DEFAULT_RUNNERS_DIR
DEFAULT_ARTIFACTS_DIR = pipeline.DEFAULT_ARTIFACTS_DIR


def _quieten_known_upstream_warnings() -> None:
    """Filter one specific, harmless upstream warning.

    torch.export emits this FutureWarning from inside pytree once per traced
    submodule, so a single export prints it dozens of times and buries the
    report. It is matched by exact message text: nothing else is silenced, and
    any warning we have not seen before still reaches the user.
    """
    # Matched loosely enough to survive the backticks in the upstream text, but
    # still tied to this one message: it must mention both treespec and LeafSpec.
    warnings.filterwarnings(
        "ignore",
        message=r".*treespec.*LeafSpec.*",
        category=FutureWarning,
    )


def run_optimize(target: str, open_report: bool = True,
                 workspace_root: str = ".", interactive: bool = True,
                 allow_ai_source: bool = False, ai_repair: bool = False,
                 runners_dir: str = None, **pipeline_options) -> int:
    """Prepare, analyze and repair the model the user pointed at.

    The report is written either way; `open_report` only decides whether a
    browser is launched. It defaults to True because a developer who just
    watched a device benchmark wants to see the result, and it is forced off
    for non-interactive runs, where a CI job opening a browser would be a
    surprise rather than a convenience.
    """
    resolved = model_source.resolve_model_input(target, workspace_root)

    verbose = bool(pipeline_options.get("verbose"))
    preflight = environment_check.preflight()
    if not preflight.ok:
        # Before exporting, lowering, uploading or benchmarking. A dependency
        # problem should cost seconds, not a whole device round trip.
        print(environment_check.format_preflight_failure(preflight, verbose),
              file=sys.stderr)
        return 2

    if runners_dir is not None:
        pipeline_options["runners_dir"] = runners_dir
        # Offered after the model is known to exist, so a typo fails on the
        # typo rather than on an Android environment the user never asked about.
        ensure_target_available(
            pipeline_options.get("target_preference", "auto"),
            interactive, runners_dir)

    pipeline_options.setdefault("interactive", interactive)
    # Two capabilities, two switches. `--allow-ai-source` authorizes sending
    # model *source* for preparation, which is a privacy boundary.
    # `--ai-repair` enables the experimental optimization-time investigation.
    # One flag for both meant opting into an experiment also authorized a
    # source upload, and vice versa.
    pipeline_options.setdefault("ai_repair", ai_repair)

    model_spec = prepare_model_source(
        resolved.path, interactive=interactive, allow_ai_source=allow_ai_source,
        verbose=verbose)

    outcome = pipeline.run_optimization(model_spec, **pipeline_options)

    if open_report and interactive:
        # Never fatal. `open_report` already handles a browser that refuses,
        # but the analysis is finished and its artifacts are written, so
        # nothing that happens while showing them may change the exit code.
        try:
            outcome.open_report()
        except Exception as error:
            print(f"(Could not open the report: {type(error).__name__})")
    return exit_code(outcome.status)


# --- preparation: deterministic first, always ---------------------------------


def prepare_model_source(model_path: Path, interactive: bool = True,
                         allow_ai_source: bool = False, announce=print,
                         prompt=input, verbose: bool = False) -> object:
    """Turn `model.py` into a ModelSpec, preferring the deterministic route.

    The order is the whole point. AI is not consulted - not even to check
    whether it is available - until the deterministic path has been tried and
    has not produced an ExportedProgram. A file that declares the interface
    and exports cleanly never touches a provider, whether or not one is
    configured.
    """
    import tempfile

    model_path = Path(model_path)
    report = model_interface.inspect_interface(model_path)

    with tempfile.TemporaryDirectory(prefix="delegate-doctor-") as workspace:
        if report.complete:
            try:
                prepared = model_interface.prepare_from_interface(
                    model_path, Path(workspace), announce=announce)
                return model_interface.model_spec_from_prepared(prepared)
            except model_interface.ModelInterfaceError as error:
                # The interface ran and export refused. The interface stays
                # authoritative: assistance adjusts it, and never goes looking
                # for a different model in the same file.
                return _after_deterministic_failure(
                    model_path, error, Path(workspace),
                    interactive=interactive, allow_ai_source=allow_ai_source,
                    announce=announce, prompt=prompt, verbose=verbose)

        announce(report.describe())
        return _prepare_with_ai(
            model_path, reason=model_interface.missing_interface_message(report),
            interactive=interactive, allow_ai_source=allow_ai_source,
            announce=announce, prompt=prompt)


def _after_deterministic_failure(model_path, error, workspace, interactive,
                                 allow_ai_source, announce, prompt,
                                 verbose: bool = False):
    """Deterministic export failed. Help *this* interface, or say why not.

    The interface was found, was executed, and built a model. Restarting
    generic preparation here would ask "which class is the model?" about a file
    that already answered - and would reject `delegate_doctor_model` for being
    a function, which is exactly what it is supposed to be.
    """
    failure = getattr(error, "failure", None)

    announce("\nPyTorch export                  FAILED")
    if verbose and failure is not None:
        # The real reason, not a category. Sanitized when it was parsed.
        announce(failure.describe())

    if failure is not None and not failure.is_export_stage:
        # A missing dependency, unavailable weights or a wrong return type is a
        # fact about the file or the environment. No provider can change it,
        # and sending the source away to be told so would waste a request.
        raise ModelSourceError(str(error))
    if failure is None and not model_interface.is_export_failure(str(error)):
        raise ModelSourceError(str(error))

    return _assist_existing_interface(
        model_path, workspace, failure, str(error),
        interactive=interactive, allow_ai_source=allow_ai_source,
        announce=announce, prompt=prompt, verbose=verbose)


def _assist_existing_interface(model_path, workspace, failure, message,
                               interactive, allow_ai_source, announce, prompt,
                               verbose):
    """AI export assistance for an interface DelegateDoctor already ran.

    A separate path from generic preparation on purpose: the model is known,
    only the export is not, and the two questions have different answers and
    different schemas.
    """
    from .agent import export_assistance, source_inspection
    from .agent.client import build_provider

    reason = (f"{message}\n"
              f"\n"
              f"The DelegateDoctor model interface is present and built a\n"
              f"model, so the remaining problem is the export itself.")

    if not (interactive or allow_ai_source):
        raise ModelSourceError(_no_ai_permission_message(reason))

    try:
        provider = build_provider(allow_ai=True)
    except AIError as error:
        raise ModelSourceError(_no_ai_available_message(reason, str(error)))

    facts = source_inspection.inspect_source(model_path)
    decision = consent.request_source_consent(
        [model_path], interactive=interactive,
        preapproved=allow_ai_source and not interactive,
        announce=announce, prompt=prompt)
    if not decision.granted:
        raise ModelSourceError(f"{reason}\n\n{decision.reason}")

    outbound = source_inspection.prepare_source_for_transmission(facts)
    safe, refusal = source_inspection.transmission_is_safe(outbound)
    if not safe:
        raise ModelSourceError(refusal)

    announce("\nAI export assistance for the existing interface...")
    outcome = export_assistance.assist_export(
        model_path=model_path,
        workspace=workspace,
        failure=failure,
        provider=provider,
        source_text=outbound,
        export=model_interface.prepare_from_interface,
        announce=announce,
        verbose=verbose,
    )

    if not outcome.succeeded:
        raise ModelSourceError(_assistance_failed_message(reason, outcome))

    announce(f"Export assistance       APPLIED")
    announce(f"Change                  {outcome.adjustment.describe()}")
    announce("PyTorch export                  PASS")
    return model_interface.model_spec_from_prepared(outcome.prepared)


def _assistance_failed_message(reason: str, outcome) -> str:
    """Which of the several things went wrong, named rather than collapsed."""
    return (
        f"{reason}\n"
        f"\n"
        f"AI export assistance did not produce an exportable configuration.\n"
        f"\n"
        f"  {outcome.reason}\n"
        f"  attempts: {len(outcome.attempts)}\n"
        f"\n"
        f"Adjust the model interface yourself, or run with --verbose to see\n"
        f"the exact export failure."
    )


def _prepare_with_ai(model_path, reason: str, interactive: bool,
                     allow_ai_source: bool, announce, prompt) -> object:
    """AI-assisted preparation, if a provider exists and the user agrees.

    This is the first point in the whole command where a provider is built or a
    credential is read. Doing it earlier would let an AI configuration problem
    break a model that never needed AI.
    """
    from .agent.client import build_provider
    from .agent.preparation import model_spec_from_outcome, prepare_model

    if not (interactive or allow_ai_source):
        raise ModelSourceError(_no_ai_permission_message(reason))

    try:
        provider = build_provider(allow_ai=True)
    except AIError as error:
        raise ModelSourceError(_no_ai_available_message(reason, str(error)))

    announce("\nPreparing model with AI...")
    outcome = prepare_model(
        model_path,
        provider=provider,
        interactive=interactive,
        # In an interactive run, `prepare_model` asks for source consent
        # itself. `--allow-ai-source` is the non-interactive equivalent of
        # answering yes, and grants nothing in a run that can still ask.
        allow_source=allow_ai_source and not interactive,
    )
    return model_spec_from_outcome(outcome)


def _no_ai_available_message(reason: str, provider_error: str) -> str:
    return (
        f"{reason}\n"
        f"\n"
        f"AI preparation is unavailable.\n"
        f"\n"
        f"{provider_error}\n"
        f"\n"
        f"Either:\n"
        f"  1. configure an AI provider, or\n"
        f"  2. add the DelegateDoctor model interface to your source:\n"
        f"\n"
        f"{model_interface.describe_interface()}"
        f"\n"
        f"Then run the same command again."
    )


def _no_ai_permission_message(reason: str) -> str:
    return (
        f"{reason}\n"
        f"\n"
        f"AI preparation could help here, but this run is non-interactive and\n"
        f"cannot ask permission to send your source to a provider.\n"
        f"\n"
        f"Either add the DelegateDoctor model interface to your source:\n"
        f"\n"
        f"{model_interface.describe_interface()}"
        f"\n"
        f"or authorize AI explicitly for this run:\n"
        f"\n"
        f"    delegate-doctor optimize <model.py> --non-interactive "
        f"--allow-ai-source"
    )


# --- setup on demand -----------------------------------------------------------


def ensure_target_available(preference: str, interactive: bool,
                            runners_dir: str, announce=print,
                            prompt=input) -> bool:
    """Offer to provision the managed Arm64 environment when it is missing.

    Onboarding should not require reading about `setup-android` first. When a
    user asks for the emulator and it is not there, the useful response is to
    offer to build it and then carry on with what they actually typed.

    The provisioning itself is `android_setup`'s - this only decides whether to
    ask, so there is one setup implementation and not two.
    """
    if preference != target_selection.PREFERENCE_EMULATOR:
        return True
    if android_setup.managed_environment_ready(Path(runners_dir)):
        return True

    announce("\nManaged Arm64 Android environment is not ready.")
    if not interactive:
        announce("Run `delegate-doctor setup-android` to provision it.")
        return False

    try:
        answer = prompt("Set it up now? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if answer not in ("", "y", "yes"):
        return False

    status = android_setup.setup_android_runners(
        project_root=Path(PROJECT_DIR),
        runners_dir=Path(runners_dir),
        interactive=True,
    )
    return status == 0


def main(argv: Optional[list] = None) -> int:
    _quieten_known_upstream_warnings()
    parser = argparse.ArgumentParser(
        prog="delegate-doctor",
        description=(
            "Find and repair expensive ExecuTorch/XNNPACK fallback operations, "
            "then keep each repair only if it is correct and faster on Arm64.\n"
            "\n"
            "  optimize model.py   analyze and repair a PyTorch model\n"
            "  setup-android       provision the Arm64 measurement environment\n"
            "\n"
            "Make a model directly analyzable by declaring two functions in it:\n"
            "\n"
            "  def delegate_doctor_model():\n"
            "      ...            # returns a torch.nn.Module\n"
            "\n"
            "  def delegate_doctor_inputs():\n"
            "      ...            # returns a tuple of example inputs\n"
            "\n"
            "With those present, preparation is deterministic and no AI is\n"
            "used. Without them, DelegateDoctor can offer to prepare the model\n"
            "using a provider you configure.\n"
            "\n"
            "The same model is also analyzable from Python:\n"
            "\n"
            "  from delegate_doctor import optimize\n"
            "  result = optimize(model, args=(example_input,))\n"
            "\n"
            "  .pte = ExecuTorch deployment artifact, DelegateDoctor's output"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    optimize = subparsers.add_parser(
        "optimize",
        help="analyze and repair a PyTorch model (model.py)",
        description=(
            "Analyze a PyTorch model on an Arm64 Android target, and repair "
            "every portable hotspot worth repairing.\n"
            "\n"
            "  delegate-doctor optimize model.py\n"
            "  delegate-doctor optimize model.py --target emulator\n"
            "  delegate-doctor optimize model.py --target device --device SERIAL\n"
            "\n"
            "A bare filename is also looked for in models/, the local workspace "
            "beside this repository.\n"
            "\n"
            "Preparation is deterministic when the source declares "
            "delegate_doctor_model() and delegate_doctor_inputs(). Otherwise "
            "DelegateDoctor offers AI-assisted preparation.\n"
            "\n"
            "After profiling, known DelegateDoctor repairs run automatically. "
            "A rule whose operator accounts for more than 0.1% of measured "
            "runtime is attempted once for the current graph, rewriting every "
            "site it matches in a single candidate. Each candidate is verified "
            "against the original model and benchmarked against the current "
            "best, and kept only if it is correct and faster; the model is "
            "re-profiled after every one that is kept.\n"
            "\n"
            "Experimental AI repair is off by default. Pass --ai-repair to "
            "enable it; it is only consulted once no known rule matches what "
            "remains."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    optimize.add_argument("target", metavar="MODEL",
                          help="model.py source defining your PyTorch model")
    optimize.add_argument("--runners-dir", default=DEFAULT_RUNNERS_DIR,
                          help="directory holding the arm64 executor_runner binaries")
    optimize.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    optimize.add_argument("--warmup", type=int, default=5,
                          help="warmup iterations per repetition")
    optimize.add_argument("--iters", type=int, default=20,
                          help="measured iterations per repetition")
    optimize.add_argument("--reps", type=int, default=1,
                          help="interleaved before/after repetitions")
    optimize.add_argument("--threads", type=int, default=4,
                          help="CPU threads used on the device")
    optimize.add_argument("--profile-iters", type=int, default=20,
                          help="iterations traced when profiling")
    # dest is explicit: the positional model argument is already called
    # "target", and argparse would otherwise let --target overwrite it.
    optimize.add_argument("--target", default="auto", dest="target_kind",
                          choices=target_selection.PREFERENCES,
                          help="which kind of Arm target to measure on")
    optimize.add_argument("--device", default=None, metavar="SERIAL",
                          dest="target_serial",
                          help="measure on this exact adb serial")
    optimize.add_argument("--non-interactive", action="store_true",
                          help="never prompt; pick a target automatically")
    # Two capabilities, two flags. One switch for both meant opting into an
    # experiment also authorized a source upload, and vice versa.
    optimize.add_argument("--ai-repair", action="store_true",
                          help="enable experimental AI repair. Off by default, "
                               "and not needed for AI model preparation or "
                               "export assistance")
    optimize.add_argument("--allow-ai-source", action="store_true",
                          help="with --non-interactive, authorize sending "
                               "model source for AI preparation")
    optimize.add_argument("--verbose", action="store_true",
                          help="print every report section, not just the summary")
    optimize.add_argument("--no-open-report", action="store_true",
                          help="do not open the HTML report in a browser when "
                               "the run finishes (it is still written)")

    checker = subparsers.add_parser(
        "check",
        help="verify this machine can run DelegateDoctor",
        description=(
            "A fast local preflight. Reports the Python stack, the ETDump "
            "analysis path, the Android SDK and - when installed - AI "
            "support.\n"
            "\n"
            "Nothing here touches the network, a device or a provider, and no "
            "API key is read.\n"
            "\n"
            "--verbose adds every version and path, which makes a stale "
            "editable install obvious."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    checker.add_argument("--verbose", action="store_true",
                         help="show versions and paths for everything checked")

    subparsers.add_parser(
        "configure-ai",
        help="choose an AI provider and model (no API key is stored)",
        description=(
            "Choose an AI provider and model for optional model preparation "
            "and repair exploration. Provider and model are saved as "
            "non-secret configuration.\n"
            "\n"
            "DelegateDoctor does not store API keys. Supply your key through "
            "the environment when you want AI:\n"
            "\n"
            "    export DELEGATE_DOCTOR_LLM_API_KEY=\"...\"\n"
            "\n"
            "AI is optional. A model that declares the DelegateDoctor model "
            "interface is analyzed without it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    setup = subparsers.add_parser(
        "setup-android",
        help="provision the Arm64 Android environment and build the runners",
        description=(
            "Prepare what DelegateDoctor needs to measure on Arm64.\n"
            "\n"
            "Plain setup targets a PHYSICAL arm64-v8a phone, and is the fast\n"
            "path:\n"
            "\n"
            "  the Android SDK Android Studio installed - discovered through\n"
            "  ANDROID_HOME, ANDROID_SDK_ROOT or the standard location for\n"
            "  this platform. DelegateDoctor never installs an SDK itself.\n"
            "  platform-tools (adb) and the pinned NDK\n"
            "  the two cross-compiled ExecuTorch runners\n"
            "  a check for an attached arm64-v8a device\n"
            "\n"
            "It does NOT download the Arm64 emulator system image.\n"
            "\n"
            "--emulator additionally provisions the managed emulator:\n"
            "\n"
            "  the emulator package and the pinned platform\n"
            "  system-images;android-35;google_apis;arm64-v8a - a LARGE\n"
            "  download, typically several gigabytes and several minutes\n"
            "  the DelegateDoctor_ARM64 AVD, where the host supports it\n"
            "\n"
            "Use it when you have no Arm64 phone to hand.\n"
            "\n"
            "Your own AVDs, Android Studio settings and shell profile are "
            "never touched. Running this first is optional - `optimize` can "
            "offer to do it when it finds the environment missing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    setup.add_argument("--runners-dir", default=DEFAULT_RUNNERS_DIR,
                       help="where the built runner binaries are installed")
    setup.add_argument("--rebuild", action="store_true",
                       help="rebuild even if the runners already exist")
    setup.add_argument("--jobs", type=int, default=10,
                       help="parallel compile jobs")
    setup.add_argument("--non-interactive", action="store_true",
                       help="never prompt; report what is missing instead")
    setup.add_argument("--yes", action="store_true",
                       help="install missing Android components without asking")
    setup.add_argument("--emulator", action="store_true", dest="setup_emulator",
                       help="also provision the managed Arm64 emulator. This "
                            "downloads a large Android system image "
                            "(system-images;android-35;google_apis;arm64-v8a)")
    setup.add_argument("--skip-emulator", action="store_true",
                       help="deprecated: the emulator is now opt-in via "
                            "--emulator, so this is the default. Kept so "
                            "existing scripts keep working")

    args = parser.parse_args(argv)

    if args.command == "check":
        report = environment_check.run()
        print(report.format(verbose=args.verbose))
        if report.ok:
            print("\nEnvironment ready.")
            return 0
        return 2

    if args.command == "configure-ai":
        return credentials.configure_interactively()

    if args.command == "setup-android":
        try:
            return android_setup.setup_android_runners(
                project_root=Path(PROJECT_DIR),
                runners_dir=Path(args.runners_dir),
                rebuild=args.rebuild,
                parallel_jobs=args.jobs,
                interactive=not args.non_interactive,
                assume_yes=args.yes,
                setup_emulator=args.setup_emulator,
                skip_emulator=args.skip_emulator,
            )
        except android_setup.SetupError as error:
            print(f"\n{error}", file=sys.stderr)
            return 2

    if args.command == "optimize":
        interactive = not args.non_interactive
        try:
            return run_optimize(
                args.target,
                open_report=not args.no_open_report,
                interactive=interactive,
                allow_ai_source=args.allow_ai_source,
                ai_repair=args.ai_repair,
                verbose=args.verbose,
                target_preference=args.target_kind,
                target_serial=args.target_serial,
                runners_dir=args.runners_dir,
                artifacts_dir=args.artifacts_dir,
                warmup_iterations=args.warmup,
                measured_iterations=args.iters,
                repetitions=args.reps,
                threads=args.threads,
                profile_iterations=args.profile_iters,
            )
        except (ModelSourceError, ExportFailed, PreparationError,
                AIError, model_interface.ModelInterfaceError) as error:
            print(f"\n{error}", file=sys.stderr)
            return 2
        except device.DeviceError as error:
            print(f"\nDevice error:\n{error}", file=sys.stderr)
            return 2
        except android_setup.SetupError as error:
            print(f"\n{error}", file=sys.stderr)
            return 2

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
