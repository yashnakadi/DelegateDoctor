# Physical-phone benchmark suite

Every checked-in example, run through DelegateDoctor on the same physical
Android phone, with the same code revision and the same settings.

## Method

```
Target        RMX2030 · arm64-v8a · Android 10 (SDK 29)
Backend       ExecuTorch 1.4.0 + XNNPACK
Runner        executor_runner_bench (event tracer OFF)
Benchmark     5 warmup + 20 measured iterations, 1 repetition
Threads       4
Profiling     20 traced iterations, executor_runner_etdump (tracer ON)
Command       delegate-doctor optimize <example> --warmup 5 --iters 20 --reps 1
```

These are **single-run device measurements**, intended to demonstrate
end-to-end behaviour consistently across the example suite rather than to
establish per-model statistical confidence. One repetition means one measured
run: no median across runs is implied or claimed.

Every example uses **random weights** and needs no network. Delegation is a
property of the graph, so trained weights would not change which operators
XNNPACK accepts.

No experimental AI repair was used. Every repair below came from the
deterministic catalog.

## Every example

| Example | Rule | Sites | Status | p50 before | p50 after | Speedup |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `dd001_softmax/deeplabv3plus.py` | DD-001 | 1 | ACCEPTED | 376.65 ms | 132.67 ms | **2.84x** |
| `dd001_softmax/pspnet.py` | DD-001 | 1 | ACCEPTED | 224.57 ms | 73.26 ms | **3.07x** |
| `dd001_softmax/interface_mnist.py` | DD-001 | 1 | ACCEPTED | 4.63 ms | 2.42 ms | **1.91x** |
| `dd001_softmax/linknet.py` | DD-001 | 1 | ACCEPTED | 258.99 ms | 163.40 ms | 1.58x |
| `dd001_softmax/fpn.py` | DD-001 | 1 | ACCEPTED | 386.79 ms | 253.26 ms | 1.53x |
| `dd001_softmax/unet.py` | DD-001 | 1 | ACCEPTED | 525.55 ms | 343.40 ms | 1.53x |
| `dd001_softmax/unetplusplus.py` | DD-001 | 1 | ACCEPTED | 625.03 ms | 424.54 ms | 1.47x |
| `dd002_alias/ghostnet.py` | DD-002 | 32 | ACCEPTED † | 721.71 ms | 499.19 ms | **1.45x** |
| `dd003_avgpool/inception_v3.py` | DD-003 | 9 | ACCEPTED | 825.18 ms | 424.19 ms | **1.95x** |
| `dd003_avgpool/densenet169.py` | DD-003 | 3 | ACCEPTED | 343.51 ms | 323.01 ms | 1.06x |
| `fully_delegated/mobilenet_v2.py` | — | — | NO_REPAIR_REQUIRED | — | — | — |
| `fully_delegated/resnext50_32x4d.py` | — | — | NO_REPAIR_REQUIRED | — | — | — |
| `no_repair/convnext_small.py` | — | — | NO_REPAIR_AVAILABLE | — | — | — |
| `ai_adapter/densenet121.py` | — | — | not benchmarked | — | — | — |

`ai_adapter/densenet121.py` declares no DelegateDoctor model interface on
purpose: it exists to demonstrate optional AI-assisted preparation. It cannot
enter the deterministic pipeline, and no remote provider call was made to
obtain a number for it, so it has no benchmark row. Inventing one would be
worse than leaving it empty.

## Correctness

Host and device verification passed for **every** candidate above, including
the one that was rejected. Backend fidelity was `OK` throughout - no model's
ExecuTorch output drifted from PyTorch beyond the 1e-5 tolerance on this phone.

† **GhostNet-100 is the one row not measured at 1 repetition.** Its baseline is
bimodal on this handset - individual samples range from 43 ms to 827 ms in the
same run - so a single repetition reports either verdict depending on which mode
it samples. It was re-measured at `--reps 5` (100 interleaved samples per
program), which gives a stable 1.42–1.45x across two runs. Every other row in
this file is a single-repetition run as described above.
[`dd002_physical_validation.md`](dd002_physical_validation.md) has the full
distribution and the eight single-repetition runs that motivated it.

## Delegation

Operator-count delegation before repair, against measured runtime delegation -
the contrast this project exists to expose:

| Example | Operator delegation | Runtime delegation before | after |
| --- | ---: | ---: | ---: |
| `inception_v3.py` | 97.1% | **53.3%** | 100.0% |
| `pspnet.py` | 94.8% | 53.0% | 99.6% |
| `linknet.py` | 99.5% | 58.3% | 100.0% |
| `deeplabv3plus.py` | 99.5% | 59.1% | 100.0% |
| `fpn.py` | 93.8% | 70.8% | 93.4% |
| `unet.py` | 96.8% | 73.5% | 93.7% |
| `unetplusplus.py` | 95.1% | 79.5% | 93.4% |
| `interface_mnist.py` | 83.3% | 95.4% | 100.0% |
| `densenet169.py` | 98.8% | 95.5% | 99.9% |
| `ghostnet.py` | 79.9% | 99.5% | 100.0% |
| `mobilenet_v2.py` | 100.0% | 100.0% | — |
| `resnext50_32x4d.py` | 100.0% | 100.0% | — |
| `convnext_small.py` | 87.9% | — | — |

Inception V3 is the clearest case: 97.1% of its operators were delegated while
only 53.3% of its runtime was, because nine `avg_pool2d` nodes out of 314
operators carried 46.7% of it.
