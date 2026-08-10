"""DelegateDoctor - find and repair expensive ExecuTorch/XNNPACK fallbacks.

A model can be almost fully delegated by operator count while a couple of
leftover operators dominate its runtime. DelegateDoctor measures which
fallbacks actually cost time, applies a known safe graph repair, checks that
the model still computes the same thing, benchmarks it on an Arm64 target, and
keeps the repair only if it is both correct and faster.

Start at `delegate_doctor.cli.run_doctor` to read the pipeline top to bottom.
"""

__version__ = "0.1.0"
