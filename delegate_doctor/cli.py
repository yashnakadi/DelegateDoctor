"""The `delegate-doctor` command line interface.

Two commands:

    optimize model.pt2 --inputs inputs.pt    analyze a serialized graph
    setup-android                            build the Arm64 runners, once

The CLI is the *artifact* interface: a serialized `ExportedProgram` plus a
serialized input tuple, with none of the caller's Python involved. That makes it
the CI-friendly route, and the one whose deserialization boundary is kept
deliberately narrow.

For a model you already have in memory, the Python API is simpler and takes
anything `torch.export` accepts:

    from delegate_doctor import optimize
    result = optimize(model, args=(example_input,))

Both routes converge on `pipeline.run_optimization`. There is no model catalog
here and no architecture DelegateDoctor knows by name: the demonstration models
live in `examples/` and are ordinary users of the public API.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

from . import android_setup, device, pipeline, pt2_input
from .api import ExportFailed
from .pt2_input import ModelInputError
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


def run_optimize(target: str, inputs: str, open_report: bool = False,
                 **pipeline_options) -> int:
    """Analyze a `.pt2` exported program against its representative inputs.

    Loading is fully validated - including one baseline execution - before any
    lowering or device work starts, so a mismatched input file costs a second
    rather than several minutes.

    The HTML report is always written; it is only *opened* when asked, because
    a CI job launching a browser would be a surprise.
    """
    outcome = pipeline.run_optimization(
        pt2_input.load_model_spec(target, inputs), **pipeline_options
    )
    if open_report:
        outcome.open_report()
    return exit_code(outcome.status)


def main(argv: Optional[list] = None) -> int:
    _quieten_known_upstream_warnings()
    parser = argparse.ArgumentParser(
        prog="delegate-doctor",
        description=(
            "Find and repair expensive ExecuTorch/XNNPACK fallback operations, "
            "then keep the repair only if it is correct and faster on Arm64.\n"
            "\n"
            "  optimize model.pt2 --inputs inputs.pt   analyze a serialized graph\n"
            "  setup-android                           build the Arm64 runners\n"
            "\n"
            "For a model you already have in Python, use the API instead - no\n"
            "artifact files needed:\n"
            "\n"
            "  from delegate_doctor import optimize\n"
            "  result = optimize(model, args=(example_input,))\n"
            "\n"
            "Runnable demonstrations live in examples/ and use exactly that API:\n"
            "\n"
            "  python examples/<model>.py\n"
            "\n"
            "  .pt2 = PyTorch ExportedProgram, the input\n"
            "  .pte = ExecuTorch deployment artifact, the output"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    optimize = subparsers.add_parser(
        "optimize",
        help="analyze an exported PyTorch model (.pt2)",
        description=(
            "Analyze a serialized torch.export.ExportedProgram. Create the two "
            "files with PyTorch:\n"
            "\n"
            "  exported = torch.export.export(model.eval(), example_inputs)\n"
            "  torch.export.save(exported, \"model.pt2\")\n"
            "  torch.save(example_inputs, \"inputs.pt\")\n"
            "\n"
            "  delegate-doctor optimize model.pt2 --inputs inputs.pt\n"
            "\n"
            "The inputs file holds the positional fp32 tensors the program is "
            "called with. It is read with weights_only=True, so it may contain "
            "only tensors and plain tuples or lists."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    optimize.add_argument("target", metavar="MODEL.pt2",
                          help="a serialized torch.export.ExportedProgram")
    optimize.add_argument("--inputs", required=True, metavar="INPUTS.pt",
                          help="torch.save()d tuple of representative fp32 tensors")
    optimize.add_argument("--runners-dir", default=DEFAULT_RUNNERS_DIR,
                          help="directory holding the arm64 executor_runner binaries")
    optimize.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    optimize.add_argument("--warmup", type=int, default=20,
                          help="warmup iterations per repetition")
    optimize.add_argument("--iters", type=int, default=150,
                          help="measured iterations per repetition")
    optimize.add_argument("--reps", type=int, default=3,
                          help="interleaved before/after repetitions")
    optimize.add_argument("--threads", type=int, default=4,
                          help="CPU threads used on the device")
    optimize.add_argument("--profile-iters", type=int, default=20,
                          help="iterations traced when profiling")
    optimize.add_argument("--verbose", action="store_true",
                          help="print every report section, not just the summary")
    optimize.add_argument("--open-report", action="store_true",
                          help="open the generated HTML report in a browser")

    setup = subparsers.add_parser(
        "setup-android",
        help="download the pinned ExecuTorch source and build the Arm64 runners",
    )
    setup.add_argument("--runners-dir", default=DEFAULT_RUNNERS_DIR,
                       help="where the built runner binaries are installed")
    setup.add_argument("--rebuild", action="store_true",
                       help="rebuild even if the runners already exist")
    setup.add_argument("--jobs", type=int, default=10,
                       help="parallel compile jobs")

    args = parser.parse_args(argv)

    if args.command == "setup-android":
        try:
            return android_setup.setup_android_runners(
                project_root=Path(PROJECT_DIR),
                runners_dir=Path(args.runners_dir),
                rebuild=args.rebuild,
                parallel_jobs=args.jobs,
            )
        except android_setup.SetupError as error:
            print(f"\n{error}", file=sys.stderr)
            return 2

    if args.command == "optimize":
        try:
            return run_optimize(
                args.target,
                args.inputs,
                open_report=args.open_report,
                verbose=args.verbose,
                runners_dir=args.runners_dir,
                artifacts_dir=args.artifacts_dir,
                warmup_iterations=args.warmup,
                measured_iterations=args.iters,
                repetitions=args.reps,
                threads=args.threads,
                profile_iterations=args.profile_iters,
            )
        except (ModelInputError, ExportFailed) as error:
            print(f"\n{error}", file=sys.stderr)
            return 2
        except device.DeviceError as error:
            print(f"\nDevice error:\n{error}", file=sys.stderr)
            return 2

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
