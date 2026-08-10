# DelegateDoctor

**Find the ExecuTorch operators that silently fall back to slow portable kernels
on Arm64, repair them, and keep the repair only if it is provably correct and
measurably faster.**

> Not all fallbacks are equal. Optimise runtime, not operator counts.

---

## The Problem

When you lower a model to ExecuTorch with the XNNPACK backend, the partitioner
accepts most operators and quietly refuses a few. The refused ones fall back to
ExecuTorch's portable reference kernels: single-threaded C++ written for
clarity, not speed.

The usual health check is *what fraction of operators got delegated?* That
number is easy to compute and easy to misread. A model can be 96.8% delegated
and still spend most of its life outside the delegate, because one refused
operator sits on a huge tensor with a terrible memory access pattern.

Operator counts weight every node equally. Hardware does not.

## What DelegateDoctor Does

```
model -> export -> XNNPACK partition -> analyse fallbacks -> profile on device
      -> rank hotspots by measured time -> detect a known pattern -> repair
      -> re-export -> verify numerically -> benchmark -> accept or reject
```

It reports **runtime-weighted delegation** — the share of wall time actually
spent inside XNNPACK, measured on the device with ETDump — and ranks every
fallback by the milliseconds it costs. When it recognises a repairable pattern
it rewrites the graph, then puts the result through two gates: outputs must
match, and latency must improve. Failing either gate discards the repair.

## Demonstrated Result

Real model, unmodified: `segmentation_models_pytorch` U-Net with a MobileNetV2
encoder, 21 classes, 256x256 input. Measured on an Arm64 Android emulator.

```
Operator-count delegation:    96.8%  ->  97.4%      +0.6 points
Runtime-weighted delegation:  35.0%  ->  93.1%

p50 latency:                76.493 ms -> 26.114 ms  2.93x (65.9% lower)

Max absolute error: 1.863e-08     Argmax agreement: 100%
REPAIR ACCEPTED
```

Operator-count delegation moved by 0.6 percentage points. A tool reporting only
operator counts would have called this model already optimised. Median latency
fell by roughly two thirds — because **one portable softmax accounted for about
63% of runtime**.

That gap is the whole point of the project.

Full recorded run: [`results/example_run.txt`](results/example_run.txt).
These are emulator numbers, not handset numbers — see [Limitations](#limitations).

## How It Works

**Profiling.** The model runs on the device under an ExecuTorch build with the
event tracer on, producing an ETDump trace read back through
`executorch.devtools.Inspector`.

**Benchmarking.** A *second*, tracer-free build measures latency, so profiling
instrumentation can never contaminate the number a decision rests on.

**Repair.** DD-001 rewrites the exported ATen graph before lowering, then the
model is re-exported and re-partitioned so the improvement is real rather than
assumed.

**Gates.** Outputs of the original and repaired programs are compared
element-wise; latency is compared on the device. Both must pass.

---

## Quick Start

### Requirements

| | |
| --- | --- |
| **Python** | **3.12** (validated on 3.12.7; enforced by `requires-python`) |
| ExecuTorch | 1.4.0, installed automatically |
| Android NDK | required by `setup-android` (tested with 27.2.12479018) |
| CMake + git | required by `setup-android` |
| Arm64 Android target | required by `doctor`, via `adb` |
| Network | required the first time `setup-android` runs |

### Install

```bash
git clone <repo-url>
cd delegate-doctor

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

That installs everything the built-in demo needs, including
`segmentation_models_pytorch`. For the test suite, use `pip install -e ".[dev]"`.

### Build Android runners

```bash
delegate-doctor setup-android
```

This downloads the exact pinned ExecuTorch source revision into the project's
own ignored `.build/` directory and cross-compiles two `arm64-v8a` runners into
`runners/`. It takes several minutes and roughly a gigabyte of download the
first time.

- **No Android device is needed for this step.** It is a build, not a run.
- **No sibling `executorch/` checkout is required.** The source is fetched into
  `.build/` automatically.
- Re-running it is safe: if both runners are present it reports that and exits.
  Use `--rebuild` to force a rebuild.

If the NDK is not found, set one of `ANDROID_NDK_HOME`, `ANDROID_NDK_ROOT` or
`ANDROID_NDK`:

```bash
export ANDROID_NDK_HOME="$HOME/Library/Android/sdk/ndk/27.2.12479018"
```

### Connect an Arm64 Android target

`doctor` executes the model on real Arm64 hardware, so it needs a target:

```bash
adb devices
adb shell getprop ro.product.cpu.abi     # must print arm64-v8a
```

- A **physical Arm64 Android phone is preferred** — it is the only way to get
  representative performance numbers.
- An **Arm64 Android emulator is acceptable** and is what the demonstrated
  result above was measured on. The system image must be `arm64-v8a`; an x86_64
  image will not work.

To create and boot an emulator:

```bash
export ANDROID_HOME=/path/to/your/android/sdk

sdkmanager --install "platform-tools" "emulator" "platforms;android-35" \
                     "system-images;android-35;google_apis;arm64-v8a"
avdmanager create avd -n dd_arm64 -k "system-images;android-35;google_apis;arm64-v8a"

# the emulator binary usually lives in the SDK rather than on PATH
"$ANDROID_HOME/emulator/emulator" -avd dd_arm64 -no-window -no-audio -no-snapshot -gpu off -cores 4 &
adb wait-for-device
```

### Run DelegateDoctor

```bash
delegate-doctor doctor unet
```

Exit code is 0 if the repair was accepted, 1 if rejected, 2 on a setup or device
error. Useful flags:

```bash
delegate-doctor doctor unet \
  --warmup 20 --iters 150 --reps 3 \   # benchmark shape
  --threads 4 \                        # device CPU threads
  --profile-iters 20 \                 # traced iterations
  --seed 1234                          # deterministic input
```

To run your own model, point `doctor` at a Python file defining `build_model()`:

```python
# my_model.py
import torch
from delegate_doctor.export_model import ModelSpec

def build_model() -> ModelSpec:
    return ModelSpec(
        name="My model",
        model=my_module.eval(),
        example_inputs=(torch.randn(1, 3, 256, 256),),
        argmax_dim=1,        # or None if an argmax is not meaningful
        description="...",
    )
```

```bash
delegate-doctor doctor my_model.py
```

## Example Output

```
DelegateDoctor

Model: U-Net / MobileNetV2
Backend: ExecuTorch + XNNPACK
Target: Arm64 Android emulator - sdk_gphone64_arm64 (arm64-v8a, Android 15)

ANALYSIS
----------------------------------------
Graph operators:             190
Delegated operators:         184
Portable operators:          6

Operator-count delegation:   96.8%
Runtime-weighted delegation: 35.0%

WARNING:
A small number of fallback operations
dominate model runtime.

FALLBACK HOTSPOTS
----------------------------------------

1. _softmax.out

Portable runtime:            48.155 ms (1 call(s))
Runtime impact:              62.7%

Known repair:
DD-001 - non-last-dimension softmax

Repair available: YES

DD-001 DETECTION
----------------------------------------
DD-001 detected

Node: softmax
Tensor rank: 4
Softmax dimension: 1
Last dimension: 3
Input shape: (1, 21, 256, 256)

VERIFICATION
----------------------------------------
Max absolute error:        1.863e-08
Argmax agreement:          100.0000%

Numerical verification: PASS

BENCHMARK
----------------------------------------
                         BEFORE      AFTER
p50 latency              76.493 ms   26.114 ms

Speedup (p50):             2.93x

DECISION
----------------------------------------
REPAIR ACCEPTED

The repaired model is numerically correct and achieves a 2.93x speedup
(65.9% lower p50 latency).
```

Each run also writes `artifacts/run_NNN/` containing both `.pte` files, readable
graphs, ETDump traces, profiles, `verification.json`, `benchmark.json`,
`results.json` and `report.txt`.

## DD-001

The one repair rule in this release.

**The problem.** ExecuTorch's XNNPACK partitioner delegates a softmax only when
its target dimension is the **last** dimension. From
`backends/xnnpack/partition/config/generic_node_configs.py`:

```python
if not (dim == -1 or dim == tensor_dims - 1):
    why(node, reason="dim must be the last dim")
    return False
```

A softmax on any other axis falls back to the portable kernel. On the U-Net's
`(1, 21, 256, 256)` tensor with `dim=1`, each of 65 536 softmax vectors has 21
members that are 65 536 elements — 256 KB — apart, so effectively every element
access is a cache miss, single-threaded.

This is not an exotic shape. It is how essentially every multi-class
segmentation model turns `(N, classes, H, W)` logits into probabilities.

**The repair.** Move the softmax axis to the end, softmax there, move it back:

```
view(A, C, B) -> permute(0, 2, 1) -> softmax(-1) -> permute(0, 2, 1) -> view(original)
```

`view`, `permute` and last-dimension `softmax` all have XNNPACK partitioner
configs, so the region rejoins the delegate. Softmax normalises independently
along one axis, so this computes the same function; in fp32 the two paths differ
only by kernel rounding.

**Why it flattens to 3-D.** The obvious 4-D version,
`x.permute(0, 2, 3, 1) -> softmax(-1) -> permute(0, 3, 1, 2)`, is mathematically
identical, fully delegates, and on ExecuTorch 1.4.0 **silently produces wrong
results** when the softmax input comes from a node XNNPACK evaluates in NHWC
layout (any convolution, or a bilinear resize). Measured on a segmentation
model: max absolute error 4.75e-02, and only 15.3% of pixels kept the correct
predicted class. 3-D tensors are not subject to that channels-last tagging.
`tests/test_dd001_rewrite.py` asserts DD-001 never emits a rank-4 permute.

**Where it runs.** DD-001 rewrites the `ExportedProgram` produced by
`torch.export.export`, *before* `to_edge_transform_and_lower`. That is the last
stage where the graph is still plain ATen operators. Once a `.pte` exists the
delegated regions are opaque compiled blobs, so **DelegateDoctor cannot repair a
`.pte` file** — it needs the model and re-exports it.

## Why Runtime-Weighted Delegation Matters

Measured, never estimated from operator types:

1. Run on the device with the event-tracer runner, producing an ETDump trace.
2. Read it with `executorch.devtools.Inspector`.
3. Sum the top-level instruction events.

Step 3 needs care, because ETDump events nest:

```
DELEGATE_CALL                       <- one whole XNNPACK blob (top level)
    Convolution (NHWC, F32) IGEMM   <- XNNPACK's internal nodes
OPERATOR_CALL                       <- one portable kernel (top level)
    native_call__softmax.out        <- that kernel's own event
Method::execute                     <- total for the inference
```

Summing every row would count the same time two or three times. DelegateDoctor
totals **only** the `DELEGATE_CALL` and `OPERATOR_CALL` rows:

```
runtime-weighted delegation = sum(DELEGATE_CALL) / (sum(DELEGATE_CALL) + sum(OPERATOR_CALL))
```

and cross-checks that against `Method::execute`, warning if they disagree by
more than 5%. Per-event times use the **median** across traced iterations, since
the runner has no warmup flag and a cold first iteration can swing a mean by
tens of percent. The nested `native_call_*` rows say *which* portable kernel
burned the time, turning a percentage into a ranked hotspot list.

## Correctness and Benchmark Gates

A repair is kept only if it passes both.

**Numerical gate** (`delegate_doctor/verification.py`). Thresholds are at the
top of the module:

```python
MAX_ABSOLUTE_ERROR_TOLERANCE = 1e-5   # ~100x fp32 epsilon
REQUIRED_ARGMAX_AGREEMENT = 1.0       # every pixel keeps its predicted class
```

**Performance gate** (`delegate_doctor/benchmarking.py`). Both `.pte` files are
benchmarked on the device under identical conditions — same input bytes, same
thread count, same runner — interleaved before/after across repetitions so drift
hits both equally.

**Decision** (`delegate_doctor/decision.py`):

```python
def decide_repair(verification_passed, before_latency_ms, after_latency_ms) -> RepairDecision
```

Note what is *not* a parameter: delegation. It is diagnostic, never an
acceptance criterion. Both rejection paths exist because both were needed for
real during development, and both are regression-tested:

- a **bit-exact** rewrite that improved every structural metric — delegate blobs
  4 -> 1, portable operators 3 -> 0, operator delegation 87.5% -> 100% — and ran
  **19% slower**;
- a rewrite that was **~30x faster**, fully delegated, and corrupted 85% of
  output pixels.

## Project Structure

```
delegate-doctor/
├── delegate_doctor/
│   ├── cli.py               the doctor / setup-android commands
│   ├── android_setup.py     fetch pinned ExecuTorch source, build runners
│   ├── export_model.py      export + XNNPACK lowering + .pte
│   ├── delegation.py        operator-count delegation
│   ├── profiling.py         ETDump -> runtime-weighted delegation, hotspots
│   ├── verification.py      the numerical gate
│   ├── benchmarking.py      on-device latency, tracer-free
│   ├── decision.py          accept / reject
│   ├── device.py            adb and runner discovery
│   ├── reporting.py         terminal report + JSON
│   └── repairs/
│       └── dd001_softmax.py DD-001 detection and rewrite
├── examples/
│   └── segmentation_unet.py the demo workload (not part of the tool)
├── tests/
├── runners/                 built by setup-android (git-ignored)
├── artifacts/               per-run output (git-ignored)
└── results/example_run.txt  recorded evidence from an earlier run
```

## Testing

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
```

The suite is fully offline: no network, no Android NDK, no emulator, no `adb`,
no ExecuTorch source checkout. Subprocess and filesystem boundaries are mocked.

- `test_dd001_detection.py` — rank-4 `dim=1` detected, rank-7 non-last detected,
  last-dim not detected, unsupported rank and dynamic shapes rejected clearly.
- `test_dd001_rewrite.py` — shapes and values preserved, softmax becomes a
  last-dim softmax, and no rank-4 permute is ever emitted.
- `test_verification.py` — rounding noise passes; a transposed output fails; a
  change too small to breach the error budget still fails if it flips a class.
- `test_decision_gate.py` — the four correct/incorrect x faster/slower
  combinations, plus regression tests for both real failures above.
- `test_android_setup.py` — version pinning, tool and NDK discovery, source
  checkout logic, runner install and verification, idempotence, CLI dispatch.

## Reproducibility

The native runners are built from an exact pinned revision, declared in
`delegate_doctor/android_setup.py`:

```python
SUPPORTED_EXECUTORCH_VERSION = "1.4.0"
EXECUTORCH_COMMIT = "3dd7ccd1d863fad22639dd2d918ae34a41ce45f0"
```

`setup-android` refuses to run if the installed ExecuTorch Python package is not
1.4.0. Building a native runtime from a different revision than the Python
package can produce wrong answers rather than an obvious failure, so it stops
rather than guessing.

Two runners are built, differing in exactly one CMake flag:

| | `EXECUTORCH_ENABLE_EVENT_TRACER` | used for |
| --- | --- | --- |
| `runners/executor_runner_etdump` | `ON` | ETDump profiling |
| `runners/executor_runner_bench` | `OFF` | latency benchmarking |

They are kept separate because the tracer adds per-instruction overhead. A
single instrumented binary would quietly corrupt every latency measurement.

## Limitations

- **One repair rule.** DD-001 only. No rule registry or plugin system, on
  purpose.
- **ExecuTorch + XNNPACK only**, fp32 only, static shapes only. DD-001 declines
  dynamic shapes rather than baking in a traced size.
- **Cannot repair a `.pte`.** The tool needs the model definition and re-exports
  it.
- **Single output tensor.** Verification compares the first output only.
- **Python 3.12 only**, and **ExecuTorch 1.4.0 only**. Both are enforced rather
  than assumed.
- **The demonstrated numbers come from an Arm64 Android emulator, not a
  handset.** Arm64 code runs natively on an Apple Silicon host, but cache sizes,
  memory bandwidth and CPU scheduling differ from a phone. The direction and
  rough scale of the effect are sound; treat the exact multiplier as provisional
  until re-measured on real hardware. A physical device works through the same
  `adb` path with no code changes.
- **Verification runs on the host**, comparing the two `.pte` files through
  ExecuTorch's Python runtime rather than on the device.
- **First `setup-android` needs network and an NDK**, and downloads roughly a
  gigabyte into `.build/`.
- **Runner architecture verification depends on the host `file` tool**; where it
  is unavailable, setup says the architecture was not checked rather than
  implying it was.
- **Not a general graph optimizer**, and not intended to become one.

## Advanced / Manual Runner Build

Most users never need this — `setup-android` automates exactly these commands.
They are kept for auditing what setup does and for building against a modified
ExecuTorch tree. The clone directory **must** be named `executorch`; the
ExecuTorch build enforces it.

```bash
git init executorch && cd executorch
git remote add origin https://github.com/pytorch/executorch.git
git fetch --depth 1 origin 3dd7ccd1d863fad22639dd2d918ae34a41ce45f0
git checkout FETCH_HEAD
git submodule update --init --recursive --depth 1

export ANDROID_NDK="$ANDROID_NDK_HOME"
export PY="$(which python)"     # the project's virtualenv interpreter

# (A) tracer-free build, used for benchmarking
cmake -S . -B cmake-out-android-bench \
  -DEXECUTORCH_BUILD_PRESET_FILE=tools/cmake/preset/android.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_PLATFORM=android-28 -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL=ON \
  -DEXECUTORCH_ENABLE_EVENT_TRACER=OFF \
  -DEXECUTORCH_BUILD_ANDROID_JNI=OFF -DEXECUTORCH_BUILD_EXTENSION_LLM=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=OFF -DEXECUTORCH_BUILD_KERNELS_LLM=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_TRAINING=OFF -DPYTHON_EXECUTABLE=$PY
cmake --build cmake-out-android-bench -j10

# (B) event-tracer build, used for profiling: same command with
#     -B cmake-out-android-etdump and -DEXECUTORCH_ENABLE_EVENT_TRACER=ON
```

The android preset keeps XNNPACK, optimized kernels and quantized kernels on,
matching what ExecuTorch ships in its Android AAR. The LLM and training
extensions are off purely to shorten the build.

Install the results under the names DelegateDoctor expects, optionally stripped:

```bash
STRIP="$ANDROID_NDK"/toolchains/llvm/prebuilt/*/bin/llvm-strip
$STRIP cmake-out-android-bench/executor_runner  -o <project>/runners/executor_runner_bench
$STRIP cmake-out-android-etdump/executor_runner -o <project>/runners/executor_runner_etdump
```

`--runners-dir` points elsewhere if you prefer.

## License

MIT. See [LICENSE](LICENSE).
