# DelegateDoctor

A diagnostic and repair tool for PyTorch models deployed to Arm64 Android through ExecuTorch and XNNPACK.

DelegateDoctor exports your model, lowers it with the XNNPACK partitioner, runs it on a real Arm64 Android target, and measures where inference time actually goes. When a known graph pattern is starving the delegate, it rewrites the graph, verifies the rewrite against the original model on both host and device, benchmarks it on the Arm target, and keeps the change only if it is correct **and** faster.

> **Not all fallbacks are equal. Optimize runtime, not operator counts.**

```bash
delegate-doctor optimize examples/dd003_avgpool/inception_v3.py --target emulator
```

```
[1] avg_pool2d.out          60.8%
    DD-003 found
    Applying to 9 matching sites...
    Host correctness       PASS
    Device correctness     PASS
    p50                    96.68 -> 36.49 ms
    Result                 ACCEPTED

Runtime delegation      39.1% -> 99.8%
Speedup                 2.65x
```

Every number above was measured on an Arm64 Android target, not estimated. Inference runs locally on the device, and the default workflow contacts no AI provider.

---

## Why DelegateDoctor

When you lower a PyTorch model to ExecuTorch with the XNNPACK backend, some operators are accepted into the delegate and some are not. The ones that are not fall back to ExecuTorch's portable kernels.

The usual way to report this is the fraction of operators that were delegated. That number is reassuring and frequently wrong.

Inception V3, measured:

| | before repair |
| --- | --- |
| Operators delegated | **97.1%** (9 of 314 fell back) |
| Runtime inside XNNPACK | **39.1%** |

Nine `avg_pool2d` nodes out of 314 operators — under 3% of the graph — accounted for **60.8% of measured inference time**. A tool reporting operator counts would have called this model well optimized and moved on.

This happens because operator count says nothing about cost. One fallback on a large tensor can outweigh hundreds of delegated operators, and each fallback also splits the delegate into separate blobs, adding layout conversions at every boundary.

DelegateDoctor therefore ranks fallbacks by **measured runtime share** taken from ExecuTorch's ETDump profiler on the Arm target, and treats operator-count delegation as diagnostic only.

**Analysis is the product; optimization is an additional capability.** A run that finds no repair is still useful: you learn where the time goes, which operators XNNPACK declined, and what the ceiling would be if they were free.

---

## What it does

```
PyTorch model (nn.Module)
        |
        v  torch.export
   ExportedProgram
        |
        v  to_edge_transform_and_lower + XNNPACK partitioner
   ExecuTorch .pte
        |
        v  run on Arm64 Android, ETDump event trace
   profile
        |
        v  rank portable fallbacks by measured runtime share
   hotspots
        |
        v  match against the repair catalog, rewrite the graph
   candidate
        |
        +--> host correctness      vs the original program
        +--> device correctness    vs the original, on the Arm target
        +--> backend fidelity      how well the backend tracks PyTorch
        +--> benchmark             p50, tracer-free runner, interleaved
              |
              +-- correct and faster ---> KEEP, re-profile, look again
              `-- otherwise ------------> REJECT
```

Repairs accumulate: after one is accepted the model is re-profiled, because removing a 60% hotspot changes every other percentage and can expose a fallback that was hidden behind it.

---

## Demonstrated results

Three rules, validated across twelve model architectures. Emulator and physical-device measurements are never mixed.

### DD-001 — physical device (RMX2030)

One unchanged rule applied to six independent `segmentation_models_pytorch` architectures. All six matched the same pattern, were repaired by the same code with no architecture-specific branches, passed host and Android verification, and got faster on a physical phone.

Target: RMX2030, Snapdragon SDM665, arm64-v8a, Android 10. Tracer-free runner, 4 threads, 20 warmup + 150 measured x 3 reps.

| Architecture | Softmax runtime | Runtime delegation | p50 before | p50 after | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| PSPNet | 38.0% | 61.9% → 99.4% | 242.69 ms | 65.53 ms | 3.703x |
| Linknet | 42.0% | 58.0% → 100.0% | 267.09 ms | 111.82 ms | 2.389x |
| DeepLabV3Plus | 38.6% | 61.4% → 100.0% | 286.29 ms | 180.38 ms | 1.587x |
| FPN | 22.8% | 73.6% → 93.1% | 394.77 ms | 253.65 ms | 1.556x |
| Unet | 20.8% | 75.1% → 94.0% | 459.61 ms | 338.10 ms | 1.359x |
| UnetPlusPlus | 18.0% | 77.2% → 93.7% | 573.52 ms | 442.84 ms | 1.295x |

Single runs, except PSPNet and Linknet, which were repeated three times: **median 3.794x** and **median 2.217x** respectively. PSPNet's spread across runs was wide (3.70–5.12x) — this is a phone under sustained load, so the median is the number to quote.

Operator-count delegation moved by 0.5–1.5 points across these six models while runtime-weighted delegation moved by 16–42 points. That gap is the entire thesis of this project.

Full record: [`results/dd001_segmentation_generalization.md`](results/dd001_segmentation_generalization.md)

### DD-003 — Arm64 Android emulator

The clearest illustration of runtime-weighted delegation. Measured on an Arm64 Android emulator (`sdk_gphone64_arm64`, arm64-v8a, Android 15) on an Apple Silicon host — **not a handset**.

| Model | Sites repaired | Operator delegation | Runtime delegation | p50 | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Inception V3 | 9 | 97.1% → 100.0% | **39.1% → 99.8%** | 96.68 → 36.49 ms | **2.65x** |
| DenseNet121 | 3 | 98.4% → 99.1% | 88.3% → 99.8% | 22.92 → 19.81 ms | 1.16x |
| DenseNet169 | 3 | 98.8% → 99.3% | 89.8% → 99.9% | 27.48 → 24.86 ms | 1.11x |

The rewrite is bit-exact: maximum absolute difference against the original is **0.000e+00** on all three models in eager PyTorch, across a sweep of nine input shapes crossed with five kernel/stride/padding combinations.

DenseNet121 and DenseNet169 matter because DD-003 was never designed against them. They hit a different form of the same rule — zero padding, where the fix is a flag change and no node is added at all.

Full record: [`results/dd003_avgpool_validation.md`](results/dd003_avgpool_validation.md)

### DD-002 — Arm64 Android emulator

Three timm GhostNet variants, bit-exact (0.000e+00 host and device), median 1.09–1.46x across three runs each; 8 of 9 runs favoured the repaired graph. The same repair on the physical RMX2030 was **inconclusive** — that phone's latency distribution was strongly bimodal (~44 ms and ~600 ms modes), and DelegateDoctor's own gate is what recorded the result as inconclusive rather than as a win.

Full record: [`results/dd002_emulator_validation.md`](results/dd002_emulator_validation.md)

A recorded end-to-end run, readable without running anything: [`results/example_run.txt`](results/example_run.txt)

---

## Repair catalog

| Rule | Pattern it recognizes | Rewrite |
| --- | --- | --- |
| **DD-001** | `softmax` on a non-last dimension, which the XNNPACK partitioner declines | `view → permute → softmax(dim=-1) → permute → view` |
| **DD-002** | A redundant `aten.alias` — an identity node that fragments the delegate | Delete the node and forward its input (1 op → 0 ops) |
| **DD-003** | `avg_pool2d` with `count_include_pad=True`, rejected unconditionally by the partitioner | `avg_pool2d(constant_pad_nd(x, p, 0.0), k, s, padding=0, count_include_pad=False)` |

Three things worth being precise about:

**Rules are graph patterns, not architecture checks.** No model name, class name, module path or node index appears anywhere in the detection or rewrite logic — a test asserts this by scanning the package source. That is why one DD-001 implementation covers six unrelated segmentation architectures, and why DD-003 matched DenseNet without being written for it.

**A rule refuses what it cannot prove.** DD-003 only fires when `count_include_pad=True`, `ceil_mode=False`, `divisor_override` is unset, shapes are static, and padding is non-negative — each with a recorded reason when it declines. `ceil_mode=True` would let a pooling window overhang the padded edge, which pre-padding cannot reproduce, so those nodes are left alone.

**A catalog match is not an acceptance.** It only means a candidate is worth measuring. The candidate then has to survive verification and beat the current model on the Arm target. DD-002 matched on the RMX2030 and was still not accepted there.

---

## Quick start

Requirements:

- **Python 3.12** (`>=3.12,<3.13` — the bound is deliberate and pip will enforce it)
- **Android Studio**, with its initial Setup Wizard completed

Android Studio's Setup Wizard is the one manual Android prerequisite. It installs the SDK that DelegateDoctor discovers via `ANDROID_HOME`, `ANDROID_SDK_ROOT`, or the standard install location for your platform. DelegateDoctor does not install an SDK of its own.

### macOS on Apple Silicon

```bash
git clone <repository-url>
cd delegate-doctor-repo
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[examples]"
delegate-doctor check
delegate-doctor setup-android
```

### Windows on Arm64 (PowerShell)

```powershell
git clone <repository-url>
cd delegate-doctor-repo
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[examples]"
delegate-doctor check
delegate-doctor setup-android
```

### Which hosts are supported

Your host architecture and your Android target architecture are separate things. DelegateDoctor always measures on an **arm64-v8a Android target**; the host only decides whether a *useful emulator* is available.

| Host | Arm64 emulator | Notes |
| --- | --- | --- |
| macOS on Apple Silicon | Supported and validated | arm64-v8a system images run natively. All emulator evidence in this README was measured here. |
| Windows or Linux on Arm64 | Implemented, **not validated** | Could in principle run arm64-v8a natively. DelegateDoctor has unit tests for the host-detection path, but it has never been exercised on real hardware. Prefer a physical device. |
| Any x86_64 host | Not available | An arm64-v8a image would be emulated instruction by instruction, so its latency would measure the emulation rather than Arm. DelegateDoctor refuses to provision an x86_64 emulator and call it an Arm target. Connect a physical arm64-v8a device. |

A physical arm64-v8a Android device works from any host.

### What the two setup commands do

`delegate-doctor check` is a fast local preflight. It reports your Python version, PyTorch, ExecuTorch, NumPy, pandas, the ETDump analysis path, `adb`, and the Android SDK, with a remedy for anything missing. It touches no network, no device and no API key.

`delegate-doctor setup-android` provisions the Arm side: it reuses the SDK Android Studio installed, adds the pinned platform (API 35), the arm64-v8a system image and NDK 27.2.12479018, creates a `DelegateDoctor_ARM64` emulator where the host supports one, and cross-compiles the two ExecuTorch runner binaries. Your own AVDs and Android Studio settings are never modified.

---

## Run your first model

A model that is already fully delegated — nothing to repair, and the analysis still tells you so:

```bash
delegate-doctor optimize examples/fully_delegated/mobilenet_v2.py --target emulator
```

A model with a large, repairable fallback:

```bash
delegate-doctor optimize examples/dd003_avgpool/inception_v3.py --target emulator
```

The first reports high runtime delegation and no repair required. The second finds nine `avg_pool2d` fallbacks worth about 60% of runtime, repairs them in one candidate, verifies it, benchmarks it, and writes an optimized `.pte`.

Use `--target device` instead of `--target emulator` to measure on a connected phone, or `--target auto` to let DelegateDoctor choose.

---

## Use your own model

Put a Python file anywhere (`models/` is a git-ignored workspace kept for exactly this) that declares two functions:

```python
import torch
from torchvision.models import resnet18


def delegate_doctor_model():
    model = resnet18(weights=None)
    model.eval()
    return model


def delegate_doctor_inputs():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 224, 224),)
```

```bash
delegate-doctor optimize model.py --target emulator
```

That is the whole contract. With those two functions present, preparation is fully deterministic and no AI is involved at any point.

DelegateDoctor executes your `model.py` to build the model. It does so in a child process whose environment has credential-shaped variables removed, which keeps API keys and cloud secrets away from model code and keeps a crash out of DelegateDoctor's own process. That is a trust boundary, not an OS sandbox: running a model file runs its imports and its side effects, so treat a `model.py` the way you would treat any other Python you are about to run.

Two optional functions are also recognized:

```python
def delegate_doctor_kwargs():          # keyword arguments for export
    return {"attention_mask": mask}

def delegate_doctor_dynamic_shapes():  # forwarded to torch.export unchanged
    return {"x": {0: torch.export.Dim("batch")}}
```

A dynamic graph is still analyzed, though a repair rule may decline to rewrite it — DD-003, for instance, requires static shapes to prove its rewrite is safe.

### Python API

If you already have the model object in memory:

```python
from delegate_doctor import optimize

result = optimize(model, args=(example_input,))
print(result.status, result.speedup)
```

`optimize()` takes your live `nn.Module`, so it needs no knowledge of how the model was built or how the checkpoint was loaded — `torch.export` captures the graph. It puts the module in eval mode for the export and restores its previous mode afterwards, and never writes to parameters or moves anything between devices.

The CLI path is the recommended way to reproduce this project's results, because a model file is self-contained.

---

## Physical Android device

Check that a device is attached and reports the right ABI:

```bash
adb devices
adb shell getprop ro.product.cpu.abi
```

The second command must print `arm64-v8a`. Then:

```bash
delegate-doctor optimize model.py --target device
```

With several devices attached, pick one with `--device SERIAL`. Every report records the exact target it measured on, and emulator and physical-device results are labelled distinctly, so the two are never confused after the fact.

---

## What you get

Each run writes a numbered directory under `artifacts/`:

| File | Contents |
| --- | --- |
| `report.html` | Self-contained local report — delegation, hotspot ranking, every gate, the decision |
| `report.txt` | The same run as plain text |
| `results.json` | Machine-readable summary |
| `repair_history.json` | Every repair attempted, its gates and its outcome, in order |
| `step_NN/verification.json` | Host and device numerical comparison for that candidate |
| `step_NN/benchmark.json` | Per-repetition latency samples |
| `step_NN/trace.etdump` | The raw ExecuTorch profiler trace |
| `optimized_model.pte` | **Only written when a repair was verified and faster** |

`report.html` is a single file with no external requests, so it can be opened offline or attached to a ticket. It opens automatically when a run finishes; `--no-open-report` suppresses that.

---

## How acceptance works

Four questions, asked in this order. Correctness is always measured against the ORIGINAL model, never against the previously accepted step, so drift cannot accumulate across several repairs.

**1. Host correctness.** Both `.pte` files run through ExecuTorch's Python runtime and their outputs are compared. Tolerance is `1e-5` maximum absolute error, plus 100% argmax agreement when the model has a class dimension. Failing here rejects the candidate immediately, with no device work.

**2. Device correctness.** Both programs run on the Arm target, their real output tensors are pulled back over `adb`, and the repaired output is compared against the original's — using the same tolerance. This catches a backend-specific problem the host cannot see. Failing here also rejects immediately, before the benchmark.

**3. Backend fidelity.** A separate question: how closely does ExecuTorch/XNNPACK reproduce PyTorch, for the **original** as well as the candidate? This is a property of the model and the backend, not of a rewrite. Inception V3's untouched original already sits 1.19e-05 from PyTorch eager before anything is changed, because fp32 reassociation accumulates across 300+ operators.

A pre-existing discrepancy is reported as a `WARNING` and does not reject anything. A discrepancy the *candidate* introduced — appearing where the original was clean, or more than 10x the original's — is a rejection. The tolerance is unchanged; only the attribution is.

**4. Benchmark.** p50 latency on the Arm target, tracer-free runner, before and after interleaved across repetitions so device drift affects both equally.

```
More delegation alone never accepts a repair.
```

Delegation percentage is not an input to the decision. Both failure modes behind that rule were observed during development and are regression-tested: a bit-exact rewrite that improved every structural metric and ran 19% *slower*, and a rewrite that was dramatically faster, fully delegated, and silently corrupted 85% of output pixels.

---

## Why runtime-weighted delegation matters

ExecuTorch's ETDump emits one event per executed instruction, and the events are **nested**. A single inference produces something like:

```
DELEGATE_CALL                     <- one whole XNNPACK blob   (top level)
    (its internal ops)
OPERATOR_CALL                     <- one portable operator    (top level)
    native_call__softmax.out      <- that kernel's own event  (nested)
```

Summing every row double-counts, because the nested rows are already inside their parents. DelegateDoctor totals only the top-level `DELEGATE_CALL` and `OPERATOR_CALL` rows:

```
runtime-weighted delegation = sum(DELEGATE_CALL) / (sum(DELEGATE_CALL) + sum(OPERATOR_CALL))
```

and cross-checks the total against `Method::execute`, so a trace that does not add up is reported rather than silently averaged. The nested `native_call_*` rows are still read — they are what identify *which* operator a portable cost belongs to, and they carry per-site costs, so a model with twenty-four layer norms can be told apart from one expensive one.

Profiling uses a separate event-tracer runner. Benchmarks use a tracer-free runner, so instrumentation can never influence a latency number.

---

## Examples

Grouped by what they demonstrate. All but one declare the DelegateDoctor model interface and run with no AI involved.

**DD-001 — non-last-dimension softmax**

```bash
delegate-doctor optimize examples/dd001_softmax/interface_mnist.py --target emulator
delegate-doctor optimize examples/dd001_softmax/unet.py --target emulator
delegate-doctor optimize examples/dd001_softmax/unetplusplus.py --target emulator
delegate-doctor optimize examples/dd001_softmax/fpn.py --target emulator
delegate-doctor optimize examples/dd001_softmax/pspnet.py --target emulator
delegate-doctor optimize examples/dd001_softmax/deeplabv3plus.py --target emulator
delegate-doctor optimize examples/dd001_softmax/linknet.py --target emulator
```

`interface_mnist.py` is the smallest — a handful of layers, no model library, fastest to run first.

**DD-002 — redundant alias**

```bash
delegate-doctor optimize examples/dd002_alias/ghostnet.py --target emulator
```

**DD-003 — avg_pool2d padding**

```bash
delegate-doctor optimize examples/dd003_avgpool/inception_v3.py --target emulator
delegate-doctor optimize examples/dd003_avgpool/densenet169.py --target emulator
```

**Already fully delegated** — the healthy case, where the useful output is the analysis

```bash
delegate-doctor optimize examples/fully_delegated/mobilenet_v2.py --target emulator
delegate-doctor optimize examples/fully_delegated/resnext50_32x4d.py --target emulator
```

**Fallbacks with no known repair** — an honest negative result

```bash
delegate-doctor optimize examples/no_repair/convnext_small.py --target emulator
```

**A model file with no interface** — `examples/ai_adapter/densenet121.py` deliberately declares neither `delegate_doctor_model()` nor `delegate_doctor_inputs()`. It exists to demonstrate the case optional AI preparation handles; see below.

The six DD-001 segmentation models need the `examples` extra (`segmentation_models_pytorch`); `ghostnet.py` additionally needs `timm`. `inception_v3.py`, `mobilenet_v2.py` and `convnext_small.py` build with pretrained weights, so their first run downloads a checkpoint from the PyTorch hub; the rest use random weights and need no network.

---

## Optional: AI-assisted preparation

The deterministic model interface above is the preferred and documented path, and nothing in the normal workflow requires AI.

Some ordinary model files do not declare it — `examples/ai_adapter/densenet121.py` is one, defining a plain `nn.Module` subclass and nothing else. For those, DelegateDoctor can optionally use an AI provider to work out how to construct and export the model, producing an adapter. Sharing your model **source** for that purpose is consented separately from enabling AI repair; agreeing to one is not agreeing to the other.

This is model *preparation*. It is not the experimental AI repair described next, and the two are separately controlled: `--ai-repair` never authorizes sending source, and consenting to send source never enables AI repair.

Consent is requested interactively and defaults to no. In a `--non-interactive` run there is no one to ask, so preparation stops unless you also pass `--allow-ai-source`.

```bash
delegate-doctor configure-ai      # choose a provider and model
```

Bring your own key. DelegateDoctor ships no key, owns no account and stores no credential; you supply one through the environment when you want AI:

```bash
python -m pip install -e ".[ai,examples]"
export DELEGATE_DOCTOR_LLM_API_KEY="..."
```

---

## Experimental: AI repair

**This is an experimental, opt-in feature. It is off by default and is not required to use DelegateDoctor.** Everything above works without it.

```bash
delegate-doctor optimize model.py --target emulator --ai-repair
```

Without `--ai-repair`, no provider is contacted and no proposal is requested, whatever the profile shows.

How it is constrained:

- **Known rules run first, always.** AI is consulted only once no catalog rule matches what remains. A hotspot a rule recognizes is never sent to a provider, even if that rule's repair was rejected — a rejection is an answer, not an invitation to guess.
- **Model-level eligibility.** Exploration is offered only when portable execution still exceeds 5% of measured runtime.
- **Bounded context.** The provider receives graph structure, shapes, dtypes and the measured profile. It does not receive weights, tensor values, checkpoints, your filesystem or environment variables.
- **A constrained proposal format.** A proposal is expressed as four operation types (`insert_aten_call`, `replace_uses`, `replace_argument`, `erase_node`) over a 24-entry ATen allowlist, applied by table lookup. There is no arbitrary Python, no `eval`, no imports, no shell and no tools.
- **The same gates as everything else.** An AI proposal becomes an ordinary candidate and meets the identical host, device, backend-fidelity and benchmark gates. There is no separate AI verification path and no reason for one.
- **Not finding anything is a normal outcome.** A provider error, an empty response, or an explicit "no safe repair" are each reported as themselves, and the run finishes with whatever deterministic repairs it already made.

```
The agent proposes.
DelegateDoctor verifies.
The Arm target decides.
```

This repository contains no measured evidence of an accepted AI repair. The deterministic rules are what the results above demonstrate.

---

## Reproducibility

- **ExecuTorch is pinned** to 1.4.0, commit `3dd7ccd1d863fad22639dd2d918ae34a41ce45f0`, which is what `setup-android` checks out and builds. `executorch==1.4.0` pulls torch 2.13.0 itself.
- **Two runner binaries, never merged.** `executor_runner_etdump` has the event tracer compiled in and is used only for profiling; `executor_runner_bench` has it off and is used only for timing. A single instrumented binary would quietly tax every latency measurement.
- **The Arm target is recorded** in every report — model, ABI, Android version, thread count, and whether it was an emulator.
- **Benchmarks are interleaved.** Default 20 warmup + 150 measured iterations x 3 repetitions, before and after alternating within each repetition, identical input bytes and thread count. p50 is reported, not the mean. Tunable with `--warmup`, `--iters`, `--reps`, `--threads`.
- **Device verification is a separate, untimed invocation** that writes output tensors; the timed benchmark writes none.

<details>
<summary>What <code>setup-android</code> installs</summary>

Into the SDK Android Studio already installed:

- `platforms;android-35`
- `system-images;android-35;google_apis;arm64-v8a`
- `ndk;27.2.12479018`
- an AVD named `DelegateDoctor_ARM64`, only where the host can run arm64-v8a natively

It then fetches the pinned ExecuTorch commit into `.build/` and cross-compiles the two runners into `runners/`. Both directories are git-ignored. `--rebuild` forces a rebuild, `--skip-emulator` builds only the runners, and `--jobs N` controls compile parallelism.

</details>

---

## Testing

```bash
python -m pip install -e ".[dev,examples]"
python -m pytest tests/ -q
```

1267 tests. The suite is fully offline: no Android device, no emulator, no NDK, no network and no API key. Device mechanisms are mocked; `torch.export`, lowering and the repair rules run for real.

---

## Limitations

- **ExecuTorch + XNNPACK only.** This is the one deployment path DelegateDoctor understands. Other backends and other runtimes are out of scope.
- **The repair catalog is small** — three rules. A model whose fallbacks are not among them gets analysis and an honest "no known repair", which is a real answer but not a faster model.
- **A device is required to accept a repair.** Correctness and speed are both measured on the target, so without one you get static analysis only.
- **Emulator performance is not handset performance.** Arm64 code runs natively on an Apple Silicon host, but cache sizes, memory bandwidth and scheduling differ from a phone. Emulator numbers demonstrate direction and magnitude, not handset latency.
- **Device verification reads the first output tensor**, as fp32. A model with several outputs is still analyzed and benchmarked; its device verification is marked unsupported rather than guessed at.
- **Python 3.12 only**, and ExecuTorch 1.4.0 exactly. Both bounds are enforced by the packaging, so a mismatch is a clear error rather than a subtly different environment.
- **No `.pte` entry point.** DelegateDoctor repairs the exported graph, and a `.pte`'s delegated regions are already compiled blobs. Point it at the PyTorch model instead.
- **Windows and Linux on Arm64 are unvalidated.** The host-detection path has unit tests, but no measurement has ever been taken on either.

---

## License

MIT. See [LICENSE](LICENSE).
