# DelegateDoctor

**If PyTorch can export your model, DelegateDoctor can inspect the exported
graph. It diagnoses the ExecuTorch/XNNPACK deployment path as far as the backend
and your device allow — and when it recognises a proven repair, it verifies and
benchmarks that repair on Arm before keeping it.**

```python
from delegate_doctor import optimize

result = optimize(model, args=(example_input,))
```

> Not all fallbacks are equal. Optimise runtime, not operator counts.
>
> Analysis is the product. Optimization is an additional capability, not a
> requirement for a useful answer.

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
nn.Module ─┐
           ├─> ExportedProgram ─> ExecuTorch lowering ─> XNNPACK partition
model.pt2 ─┘                                                   │
                                                               v
                              profile on device ─> rank hotspots by measured time
                                                               │
                                                               v
                                        detect a known pattern ─> repair
                                                               │
                                                               v
                             verify on host ─> verify on device ─> benchmark
                                                               │
                                                               v
                                                        accept or reject
```

It reports **runtime-weighted delegation** — the share of wall time actually
spent inside XNNPACK, measured on the device with ETDump — and ranks every
fallback by the milliseconds it costs. When it recognises a repairable pattern
it rewrites the graph, then puts the result through three gates: host outputs
must match, the tensors the **Android device actually produced** must match, and
latency must improve. Failing any gate discards the repair.

Every stage reports its own outcome, so a model that exports but cannot be
lowered, or lowers but cannot run on the attached target, still gets whatever
analysis is possible — and the report says exactly where it stopped rather than
calling the model unsupported.

## Demonstrated Result

Real model, unmodified: `segmentation_models_pytorch` U-Net with a MobileNetV2
encoder, 21 classes, 256x256 input. Measured on an Arm64 Android emulator.

```
Operator-count delegation:    96.8%  ->  97.4%      +0.6 points
Runtime-weighted delegation:  34.3%  ->  93.2%

p50 latency:                77.545 ms -> 26.853 ms  2.89x (65.4% lower)

Host verification:   max abs error 1.863e-08, argmax agreement 100%
Android verification: max abs error 1.863e-08, argmax agreement 100%
REPAIR ACCEPTED
```

Operator-count delegation moved by 0.6 percentage points. A tool reporting only
operator counts would have called this model already optimised. Median latency
fell by roughly two thirds — because **one portable softmax accounted for 63.4%
of runtime**.

That gap is the whole point of the project.

Full recorded run: [`results/example_run.txt`](results/example_run.txt).
These are emulator numbers, not handset numbers — see [Limitations](#limitations).

### The same rule across six architectures

On a physical **RMX2030** (Snapdragon 665, arm64-v8a, Android 10), the *same*
unchanged DD-001 rule was applied to six independent segmentation
architectures. All six passed host and Android verification with 100% argmax
agreement:

| Architecture | Softmax runtime | Runtime delegation | p50 before → after | Speedup | Runs |
| --- | ---: | ---: | ---: | ---: | :-: |
| PSPNet | 38.0% | 61.9% → 99.4% | 242.69 → 65.53 ms | **3.794x** | median of 3 |
| Linknet | 42.0% | 58.0% → 100.0% | 267.09 → 111.82 ms | **2.217x** | median of 3 |
| DeepLabV3+ | 38.6% | 61.4% → 100.0% | 286.29 → 180.38 ms | 1.587x | single run |
| FPN | 22.8% | 73.6% → 93.1% | 394.77 → 253.65 ms | 1.556x | single run |
| U-Net | 20.8% | 75.1% → 94.0% | 459.61 → 338.10 ms | 1.359x | single run |
| U-Net++ | 18.0% | 77.2% → 93.7% | 573.52 → 442.84 ms | 1.295x | single run |

**Only PSPNet and Linknet are repeated medians** (3 runs each); the other four
are single runs and should be read as indicative. The p50 columns show the
first run in every case.

The honest claim is not "DD-001 makes segmentation models faster". It is that
DD-001 recognises the same backend-hostile class-softmax pattern across six
independent architectures and repairs them all with one rule, with a benefit
that depends on how expensive that fallback is in each model. Full method and
caveats: [`results/dd001_segmentation_generalization.md`](results/dd001_segmentation_generalization.md).

## How It Works

**Profiling.** The model runs on the device under an ExecuTorch build with the
event tracer on, producing an ETDump trace read back through
`executorch.devtools.Inspector`.

**Benchmarking.** A *second*, tracer-free build measures latency, so profiling
instrumentation can never contaminate the number a decision rests on.

**Repair.** A rule rewrites the exported ATen graph before lowering, and the
repaired graph is then lowered and re-partitioned from scratch, so the
improvement is measured rather than assumed. The original model is never
re-traced — everything happens on the `ExportedProgram`.

**Gates.** Outputs are compared element-wise on the host *and* on the Android
device, using tensors pulled back over `adb`; latency is compared on the device.
All must pass. A rewrite can be correct on the host and still hit a
backend-specific bug, which is why the device check is part of the gate rather
than a diagnostic.

---

## Quick Start

### Requirements

| | |
| --- | --- |
| **Python** | **3.12** (validated on 3.12.7; enforced by `requires-python`) |
| ExecuTorch | 1.4.0, installed automatically |
| Android NDK | required by `setup-android` (tested with 27.2.12479018) |
| CMake + git | required by `setup-android` |
| Arm64 Android target | required for profiling and benchmarking, via `adb` |
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

That installs DelegateDoctor and its core dependencies. The library itself
needs no model zoo — `segmentation_models_pytorch`, `timm` and `torchvision` are
only used by the demonstration scripts:

```bash
python -m pip install -e ".[examples]"    # to run examples/
python -m pip install -e ".[dev]"         # to run the test suite
```

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

DelegateDoctor executes the model on real Arm64 hardware, so profiling and
benchmarking need a target:

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

### Optimize your own model

Hand DelegateDoctor the model object you already have in memory:

```python
from delegate_doctor import optimize

result = optimize(model, args=(example_input,))

result.open_report()             # opens report.html in your browser
```

The model executes and is measured on the Arm64 Android target. The report is
generated **locally** and opens in your desktop browser; nothing is ever
displayed on the phone, which is only the measurement target.

The terminal stays short:

```
DelegateDoctor - PSPNet

Result                  REPAIR ACCEPTED
Top hotspot             _softmax.out · 38.0% runtime
Runtime delegation      61.9% -> 99.4%
Latency                 242.69 -> 65.53 ms
Speedup                 3.79x
Correctness             PASS host / PASS device
Repair applied          DD-001

Optimized model         artifacts/run_042/optimized_model.pte
Report                  artifacts/run_042/report.html
```

Everything else is in the result object and the run directory:

```python
print(result.status)             # e.g. REPAIR_ACCEPTED
print(result.repair_available)   # did a catalog rule match?
print(result.output_pte)         # path to the optimized .pte, or None
print(result.report_path)        # path to report.html
```

It does not need to know how the model was built, how the checkpoint was
loaded, or how your project is laid out — `torch.export` answers all of that by
capturing the graph. Your architecture, your configuration, your trained
weights, exactly as they are:

```python
model = MyModel(...)
model.load_state_dict(torch.load("weights.pt", weights_only=True))

result = optimize(model, args=(sample,))
```

Arguments mirror `torch.export.export`, so keyword inputs and dynamic shapes
work the same way:

```python
optimize(model, args=(input_ids,), kwargs={"attention_mask": mask})

batch = torch.export.Dim("batch", min=1, max=8)
optimize(model, args=(x,), dynamic_shapes=({0: batch},))
```

**Your model object is not disturbed.** Export needs inference mode, so
DelegateDoctor switches to `eval()` and switches back afterwards. It never
writes to parameters or buffers, never moves your model between devices (it
refuses and tells you to pass a CPU copy instead), and reads through a
`DataParallel` wrapper rather than unwrapping it in place. Calling `optimize()`
in the middle of a training script does not change what that script does next.

### How far a model gets

`torch.export.export()` is the acceptance boundary. If PyTorch can capture your
model, DelegateDoctor accepts the graph and analyzes it as far as ExecuTorch,
XNNPACK and the attached Arm target allow — then tells you exactly where it
stopped:

```
PIPELINE
----------------------------------------
PyTorch export              PASS
Graph inspection            PASS
ExecuTorch lowering         PASS
XNNPACK analysis            PASS
  1 portable of 41 ops
Android execution           UNSUPPORTED
  input 0 is torch.int64; the Android input transport writes raw fp32
  blobs with no dtype header
Runtime profiling           NOT RUN
Repair matching             PASS
  DD-001 matched; not applied without a device benchmark
Correctness verification    NOT RUN
Device benchmark            NOT RUN

RESULT
----------------------------------------
DEVICE_EXECUTION_UNSUPPORTED
Static analysis complete. The Arm target could not run this model.
```

That is a **successful analysis**, not a failure. These are all separate things,
and DelegateDoctor keeps them separate:

| | |
| --- | --- |
| model exportable | `torch.export` captured the graph |
| ExecuTorch-lowerable | ExecuTorch turned it into a runnable program |
| runnable on the current Arm runner | inputs/outputs fit the device transport |
| profileable | ETDump measured where the runtime went |
| repair available | a catalog rule recognised a pattern |
| repair correct | host and device outputs still match |
| repair faster | the target benchmark improved |

`result.status` is one of:

| Status | Meaning |
| --- | --- |
| `REPAIR_ACCEPTED` | correct **and** faster on the device; `.pte` written |
| `REPAIR_REJECTED` | a gate said no — the only non-zero CLI exit |
| `NO_REPAIR_AVAILABLE` | hotspots ranked, but no rule matches them |
| `NO_REPAIR_REQUIRED` | no portable hotspot to repair |
| `FULLY_DELEGATED` | XNNPACK took every operator |
| `ANALYSIS_COMPLETE` | static analysis done; device stages did not run |
| `EXECUTORCH_LOWERING_UNSUPPORTED` | exported fine, ExecuTorch declined it |
| `DEVICE_EXECUTION_UNSUPPORTED` | lowered fine, the target could not run it |

Only `torch.export` failing raises, because then there is no graph to analyze:

```
PYTORCH EXPORT FAILED

DelegateDoctor could not capture this model as a torch.export ExportedProgram.
The model has not entered the DelegateDoctor analysis pipeline.
```

A repair is still only kept when it is **correct and measurably faster on your
target**. Nothing about a graph looking cleaner, being more delegated, or having
fewer partitions can accept a repair, and a repair that cannot be verified is
never accepted.

### The `.pt2` artifact path

The same pipeline also takes a serialized graph, which is the reproducible,
CI-friendly form — useful when optimization runs outside the process that
trained the model:

```python
exported_program = torch.export.export(model.eval(), example_inputs)
torch.export.save(exported_program, "model.pt2")
torch.save(example_inputs, "inputs.pt")
```

```bash
delegate-doctor optimize model.pt2 --inputs inputs.pt
```

```
PyTorch nn.Module ──torch.export.export()──┐
                                           ├──> ExportedProgram ──> DelegateDoctor
model.pt2 ────────torch.export.load()──────┘                             │
                                                                         v
                                                        optimized ExecuTorch .pte
```

Both routes converge on the same `ExportedProgram` and the same pipeline; there
is no second optimization engine. You do **not** need to create a `.pt2` to use
the Python API.

```
.pt2 = PyTorch ExportedProgram   — the input to DelegateDoctor
.pte = ExecuTorch program        — the output it produces
```

A `.pte` cannot be optimized. Its delegated regions are already compiled blobs,
so there is no ATen graph left to repair — re-export from PyTorch instead.

`examples/export_to_pt2.py` builds both files from a model class in a Python
file, if you want a pair to try:

```bash
python examples/export_to_pt2.py --source examples/mnist.py \
    --model Net --input-shape 1,1,28,28
```

#### The inputs file

`inputs.pt` holds the positional arguments the exported program is called with:

```python
torch.save((torch.randn(1, 3, 224, 224),), "inputs.pt")
```

They are *representative* inputs, used for three jobs — host verification,
device verification, and the benchmark — so the same tensors serve all three and
every comparison is exact.

The artifact path is deliberately **narrower than the Python API**: it accepts a
tuple (or list) of positional fp32 tensors, or a single tensor. The Python API
can accept anything `torch.export` accepts because it holds the real objects;
`inputs.pt` has to be deserialized from disk, and that safety boundary is worth
keeping tight.

> **Deserialization.** `inputs.pt` is read with
> `torch.load(..., weights_only=True)`, which restricts unpickling to tensors
> and plain containers; an artifact holding custom objects is refused rather
> than executed, and there is no fallback to unrestricted loading. `.pt2` is
> read with `torch.export.load`, PyTorch's supported deserializer — still a
> deserializer, so load `.pt2` files from sources you trust.

Current device-stage limits (none of which stop the analysis):

- the Android transport carries **positional fp32 tensors**; other dtypes and
  keyword arguments are reported as an unsupported device stage
- device verification reads back the **first output tensor**, so a model with
  several outputs is analyzed and its verification honestly marked unsupported
- your own models report tensor error only — no argmax/top-1 claim is made,
  because DelegateDoctor does not know your output semantics

## Try the examples

Build the Android runners once, and connect an Arm64 target:

```bash
delegate-doctor setup-android

adb devices
adb shell getprop ro.product.cpu.abi     # must print arm64-v8a
```

Then run any example directly:

```bash
python examples/unet.py
python examples/unetplusplus.py
python examples/fpn.py
python examples/pspnet.py
python examples/deeplabv3plus.py
python examples/linknet.py
python examples/ghostnet.py           # DD-002 demonstration
python examples/mobilenet_v2.py       # a healthy model: nothing to repair
```

They need the demo model libraries:

```bash
python -m pip install -e ".[examples]"
```

**Each script is an ordinary user of the public API.** Open any of them and the
whole file is:

```python
from delegate_doctor import optimize

model = build_model()      # a stock smp / timm / torchvision model
model.eval()

result = optimize(model, args=(example_input,))
```

There is no model-specific execution path inside DelegateDoctor. The core
package contains no dispatch for U-Net, PSPNet, GhostNet or any other example
model — it does not know they exist. `python examples/pspnet.py` is just
*construct PSPNet → `optimize(model, args=(x,))`*, which is why the six
segmentation examples are evidence that DD-001 is a real pattern rule rather
than six hard-coded special cases. A regression test asserts the core has no
architecture names in it.

The six segmentation models all produce the DD-001 pattern naturally through the
documented `activation="softmax2d"` option, each reaching it via its own
decoder. GhostNet reaches DD-002 through timm's own `GhostModule` slice. Nothing
is planted, and every run measures the connected device — nothing is
pre-computed.

`examples/mobilenet_v2.py` is the control: a mainstream mobile architecture
XNNPACK takes completely, which should report `FULLY_DELEGATED` or
`NO_REPAIR_REQUIRED` and produce no artifact. It is the one example that
downloads pretrained weights (its own choice — DelegateDoctor never downloads
anything); pass `weights=None` to stay offline.

### Tuning a run

Both the API and the CLI take the same benchmark controls:

```python
optimize(model, args=(x,), warmup_iterations=20, measured_iterations=150,
         repetitions=3, threads=4, profile_iterations=20)
```

```bash
delegate-doctor optimize model.pt2 --inputs inputs.pt \
  --warmup 20 --iters 150 --reps 3 --threads 4 --profile-iters 20
```

CLI exit code is 0 for any completed analysis, 1 if a repair was rejected, and 2
on a setup or input error.

## The report

Every run writes a self-contained `report.html` next to its other artifacts:

```
artifacts/run_042/
├── report.html          concise visual summary, opens in any browser
├── report.txt           the same run, section by section
├── results.json         every measured number
├── verification.json    host and device numerical comparison
├── benchmark.json       raw per-repetition latencies
├── optimized_model.pte  written only when a repair was accepted
├── before/  after/      .pte, readable graphs, ETDump traces, profiles
└── input0.bin           the exact bytes the device was fed
```

The report answers five questions on the first screen: is the deployment
healthy, where is runtime going, did DelegateDoctor recognise a repair, did
correctness survive, and did the target actually get faster. Its centrepiece is
operator-count delegation shown beside runtime-weighted delegation, because that
is where the two most often disagree.

It is **fully self-contained** — all CSS inline, no JavaScript, no fonts, no
images, no network. `file:///.../report.html` is enough, and it survives being
emailed to a colleague. Deeper material (all portable operators, benchmark
method, ExecuTorch version, artifact paths, patterns a rule declined) sits in a
collapsed *Technical details* section so the first screen stays short.

Open it however you like:

```python
result.open_report()          # from Python
print(result.report_path)     # or just take the path
```

```bash
delegate-doctor optimize model.pt2 --inputs inputs.pt --open-report
```

`optimize()` never opens a browser on its own — a CI job launching one would be
a surprise. The examples call `open_report()` explicitly, which is what makes
`python examples/pspnet.py` finish by showing you the result.

Add `--verbose` (CLI) or `verbose=True` (Python) to print every section to the
terminal as well; `quiet=True` silences it entirely.

Normal runs also suppress a short allowlist of known-benign PyTorch/ExecuTorch
diagnostics (a pytree deprecation, an ETDump debug-buffer notice, a CPU probe)
so the console stays readable. Unknown warnings and errors are never suppressed,
and `--verbose` restores everything. See `delegate_doctor/console_noise.py` for
the exact list and the reason each entry is safe to hide.

## Example Output

The console after a run that found and kept a repair:

```
DelegateDoctor - U-Net / MobileNetV2

Result                  REPAIR ACCEPTED
Top hotspot             _softmax.out · 62.7% runtime
Runtime delegation      35.0% -> 93.2%
Latency                 77.55 -> 26.85 ms
Speedup                 2.89x
Correctness             PASS host / PASS device
Repair applied          DD-001

Optimized model         artifacts/run_001/optimized_model.pte
Report                  artifacts/run_001/report.html
```

The same run, section by section, is in `report.txt`, and visually in
`report.html`. A full recorded example - operator counts, hotspot ranking,
verification metrics, per-repetition benchmark numbers and the decision - is
checked in at [`results/example_run.txt`](results/example_run.txt).

## Security

DelegateDoctor has two input boundaries, and they have different properties.

The **Python API** runs inside your own process, on an object you already
constructed. It reads nothing from disk and deserializes nothing; the trust
question is simply whether you trust the model you are holding.

The **`.pt2` + `inputs.pt` artifact path** does deserialize, so that is where
the restrictions live — and they stay narrow deliberately, even though the
Python API is broader.

- **No AI, no accounts, no telemetry.** There is no LLM, no cloud service and no
  API key anywhere in the tool. Analysis and optimization are entirely
  deterministic. The one thing that reaches the network is
  `delegate-doctor setup-android`, which fetches the pinned ExecuTorch source
  the first time you build the Arm64 runners.
- **No repository ingestion.** DelegateDoctor does not clone, fetch or inspect
  repositories. Remote inputs of any kind are rejected with a clear error.
- **Inputs are loaded restrictively.** `inputs.pt` is read with
  `torch.load(..., weights_only=True)`, so unpickling is limited to tensors and
  plain containers. An artifact containing custom objects is refused rather than
  executed, and there is **no fallback to unrestricted loading** — a test
  asserts the flag is never set any other way.
- **Everything is validated before anything expensive happens.** The `.pt2` must
  load as an `ExportedProgram`, produce a callable module, accept the supplied
  inputs and execute once — all before lowering, profiling or any device work.
- DelegateDoctor **never installs dependencies** and **never downloads models,
  checkpoints or inputs** automatically.

### What this is not

`torch.export.load` is PyTorch's supported deserializer for `.pt2`, and it is
still a deserializer: it reads a serialized graph and its constants from a file.
That is a much smaller surface than executing arbitrary Python — there is no
model source being imported any more — but it is not zero. Load `.pt2` files
from sources you trust, the same way you would treat any model checkpoint.

## How it fits together

```
   nn.Module + args/kwargs                     model.pt2 + inputs.pt
            |                                            |
            v  torch.export.export()                     v  torch.export.load()
            +--------------> ExportedProgram <-----------+
                             (pristine baseline)
                                    |
                     +--------------+--------------+
                     |                             |
                     v                             v
              baseline execution           ExecuTorch lowering
                     |                        /          \
                     |                    fail            pass
                     |                     |               |
                     |                     v               v
                     |            report limitation   XNNPACK analysis
                     |                                     |
                     |                                     v
                     |                          device / profile, if supported
                     |                                     |
                     |                                     v
                     |                               Repair catalog
                     |                          |- DD-001  non-last-dim softmax
                     |                          |- DD-002  redundant no-op alias
                     |                          `- future community repairs
                     |                                /          \
                     |                             none          match
                     |                              |               |
                     |                              v               v
                     |                      NO REPAIR AVAILABLE  rewrite
                     |                                              |
                     +--> host + Android verification <-------------+
                                    |    (against the pristine baseline)
                                    v
                          benchmark on YOUR target
                                    |
                          +--- slower --> REJECT
                          |
                          v faster
                     optimized_model.pte
```

A repair rule is a **known safe optimization candidate, not a promise of
universal speedup.** DelegateDoctor benchmarks every repair on your actual
target before accepting it — the same bit-exact DD-002 repair measured 1.46x on
one Arm64 target and was un-measurable on another.

## Repair catalog

Two accepted rules. Entering the catalog means a rule is correct and measurably
faster on at least one supported Arm64 Android target — **not** that it is
faster everywhere. DelegateDoctor always benchmarks original vs repaired on your
device and rejects the repair if it does not win there.

| Rule | Pattern | Repair | Validated on |
| --- | --- | --- | --- |
| **DD-001** | softmax on a non-last dimension | axis canonicalization: `view → permute → softmax(-1) → permute → view` | 6 segmentation architectures, physical RMX2030, 1.30–3.79x |
| **DD-002** | redundant `aten.alias` (a no-op that fragments the graph) | delete the node, forward its input (1 op → 0 ops) | 3 timm GhostNet variants, Arm64 **emulator**, 1.09–1.46x; physical phone inconclusive |

DD-001 fixes an unsupported *configuration* of a real operator. DD-002 removes
an operator that does nothing but split the delegate. Details:
[`results/dd002_emulator_validation.md`](results/dd002_emulator_validation.md).

## DD-001

The first of the two repair rules, and the one with the broadest evidence.

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
`.pte` file** — it needs an `ExportedProgram`, from `torch.export.export` on a
live model or `torch.export.load` on a `.pt2`.

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

Correctness is checked **twice**, on the host and on the Android device.

**Host gate** (`delegate_doctor/verification.py`). Both `.pte` files run through
ExecuTorch's Python runtime and their outputs are compared. Thresholds are at the
top of the module:

```python
MAX_ABSOLUTE_ERROR_TOLERANCE = 1e-5   # ~100x fp32 epsilon
REQUIRED_ARGMAX_AGREEMENT = 1.0       # every pixel keeps its predicted class
```

**Device gate** (`delegate_doctor/device_verification.py`). A graph rewrite can
be mathematically equivalent and still trigger a device- or backend-specific
correctness bug — the Android build of XNNPACK is different compiled code on a
different architecture. DelegateDoctor therefore runs both `.pte` files on the
Arm64 target, pulls the real output tensors back with `adb`, and checks them
using the same thresholds:

- repaired vs original, both measured **on the device** — did the repair change
  the answer on real hardware?
- each model's device output vs its own host output — this separates "the repair
  is wrong" from "the backend is wrong".

This is a separate, untimed invocation. The timed benchmark still runs with
`--print_output none` and writes no tensors, so output capture cannot pollute
latency numbers.

**Performance gate** (`delegate_doctor/benchmarking.py`). Both `.pte` files are
benchmarked on the device under identical conditions — same input bytes, same
thread count, same runner — interleaved before/after across repetitions so drift
hits both equally.

**Decision** (`delegate_doctor/decision.py`):

```python
def decide_repair(
    host_verification_passed,
    device_verification_passed,
    before_latency_ms,
    after_latency_ms,
) -> RepairDecision
```

A repair is accepted only when **host verification passes, Android verification
passes, and p50 latency improves**. A repair that improves delegation, runs
nearly 3x faster and verifies on the host is still rejected if the tensors the
device produced do not verify.

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
│   ├── __init__.py          the public API: optimize(), OptimizationResult
│   ├── api.py               live nn.Module -> torch.export -> pipeline
│   ├── pipeline.py          the one stage machine every entry point runs
│   ├── result.py            stage outcomes and the structured result
│   ├── capabilities.py      what the Android transport can carry today
│   ├── cli.py               optimize (.pt2) / setup-android
│   ├── pt2_input.py         load and validate model.pt2 + inputs.pt
│   ├── android_setup.py     fetch pinned ExecuTorch source, build runners
│   ├── export_model.py      ModelSpec, XNNPACK lowering, .pte
│   ├── delegation.py        operator-count delegation
│   ├── profiling.py         ETDump -> runtime-weighted delegation, hotspots
│   ├── verification.py      the host numerical gate
│   ├── device_verification.py  pull Android tensors and check them
│   ├── benchmarking.py      on-device latency, tracer-free
│   ├── decision.py          accept / reject
│   ├── device.py            adb and runner discovery
│   ├── reporting.py         terminal report + JSON
│   └── repairs/
│       ├── dd001_softmax.py    DD-001 detection and rewrite
│       └── dd002_noop_alias.py DD-002 detection and rewrite
├── examples/                standalone scripts; ordinary users of optimize()
│   ├── unet.py  unetplusplus.py  fpn.py           DD-001 demonstrations
│   ├── pspnet.py  deeplabv3plus.py  linknet.py
│   ├── ghostnet.py                                DD-002 demonstration
│   ├── mobilenet_v2.py                            healthy baseline
│   └── export_to_pt2.py     make model.pt2 + inputs.pt from a model class
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

The suite is fully offline and deterministic: no network, no Android NDK, no
emulator, no `adb`, no ExecuTorch source checkout, no AI of any kind. Subprocess
and filesystem boundaries are mocked.

- `test_live_api.py` — `from delegate_doctor import optimize`; the caller's
  training mode, parameters and DataParallel wrapper survive untouched; multiple
  args, kwargs and `dynamic_shapes` reach `torch.export`; export failure is
  framed as an export failure; int64 inputs and multi-output models are analyzed
  rather than rejected; the live and `.pt2` paths produce equivalent specs and
  identical repair detection.
- `test_pipeline_stages.py` — every stage outcome: a mocked lowering failure
  becomes `EXECUTORCH_LOWERING_UNSUPPORTED` (and does not blame `torch.export`),
  a missing device leaves static analysis intact and applies no repair, an
  unsupported input dtype or unverifiable output blocks acceptance without
  faking a PASS, plus the fully-delegated and no-repair-available outcomes.
- `test_pt2_input.py` — the artifact boundary: valid `.pt2` round-trips to an
  `ExportedProgram`; URLs, directories, missing files, wrong suffixes, a `.pte`
  passed as a `.pt2`, FIFOs and corrupt archives are each refused with their own
  message. Inputs: tensor tuples, lists and bare tensors accepted; dicts,
  non-tensors, empty tuples, non-fp32 and non-finite values rejected. Also
  asserts `weights_only=True` is how inputs are loaded and that no unrestricted
  fallback exists.
- `test_pipeline_boundary.py` — the pristine baseline survives a repair, the
  repaired graph really did change, lowering does not disturb the baseline, both
  rules still accept a `.pt2`-loaded program, and the CLI surface removed in the
  move to `.pt2` (the old AI-setup subcommand, its flags, and repository URLs)
  no longer resolves.
- `test_dd001_detection.py` — rank-4 `dim=1` detected, rank-7 non-last detected,
  last-dim not detected, unsupported rank and dynamic shapes rejected clearly.
- `test_dd001_rewrite.py` — shapes and values preserved, softmax becomes a
  last-dim softmax, and no rank-4 permute is ever emitted.
- `test_dd002_noop_alias.py` — the no-op alias rule, including the cases it
  declines.
- `test_verification.py` — rounding noise passes; a transposed output fails; a
  change too small to breach the error budget still fails if it flips a class.
- `test_decision_gate.py` — the four correct/incorrect x faster/slower
  combinations, plus regression tests for both real failures above.
- `test_device_verification.py` — binary tensor parsing (truncated, empty,
  oversized, wrong dtype), adb command construction with the selected serial,
  distinct before/after filenames, and the device checks that reject a wrong
  Android tensor. Includes a guard that the timed benchmark never gains
  `--output_file`.
- `test_android_setup.py` — version pinning, tool and NDK discovery, source
  checkout logic, runner install and verification, idempotence, CLI dispatch.
- `test_examples.py` — every example is a standalone script that imports the
  public `optimize`, calls it with `args=`, and reaches past nothing private;
  and the core package has no architecture names in its code, no model-name
  dispatch, no demo-library imports, and no `doctor` subcommand.
- `test_reporting_output.py` — the concise console output.

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

- **Two repair rules.** DD-001 and DD-002. `ALL_RULES` is a plain list — no rule
  registry or plugin system, on purpose.
- **ExecuTorch + XNNPACK only.** DelegateDoctor analyzes this one deployment
  path, and ExecuTorch may still decline a graph PyTorch exported happily.
- **Cannot repair a `.pte`.** Its delegated regions are already compiled blobs.
  Export a `.pt2` from PyTorch instead.
- **The device stages are narrower than the analysis.** The Android transport
  carries positional fp32 tensors, and device verification reads back the first
  output tensor. Other dtypes, keyword arguments and richer output structures
  are analyzed and then honestly reported as unsupported device stages — they no
  longer stop a model at the door, but they do stop it short of a benchmark.
- **A repair needs the device.** Detection is static, but acceptance requires
  host correctness, device correctness and a faster device benchmark. With no
  Arm64 target attached, a matched rule is reported as a candidate and nothing
  is applied.
- **DD-001 declines dynamic shapes.** A dynamic graph is still exported,
  lowered and analyzed; the rule simply will not rewrite it rather than baking
  in a traced size.
- **Your models get no argmax check.** The demo catalog knows its own output
  semantics and checks class agreement; a model DelegateDoctor has never seen is
  verified on tensor error alone.
- **The graph is fixed at export time.** DelegateDoctor analyzes exactly the
  graph `torch.export` captured. If the export took the wrong branch, shape or
  configuration, the tool has no way to know — re-export instead.
- **Python 3.12 only**, and **ExecuTorch 1.4.0 only**. Both are enforced rather
  than assumed.
- **The demonstrated numbers come from an Arm64 Android emulator, not a
  handset.** Arm64 code runs natively on an Apple Silicon host, but cache sizes,
  memory bandwidth and CPU scheduling differ from a phone. The direction and
  rough scale of the effect are sound; treat the exact multiplier as provisional
  until re-measured on real hardware. A physical device works through the same
  `adb` path with no code changes.
- **Device verification covers the first output tensor, fp32 only.** The Android
  runner writes raw bytes with no dtype tag, so the expected dtype, shape and
  byte count come from the host result and anything else is rejected rather than
  reinterpreted.
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
