# DD-001 on a physical phone — result: GENERALIZES

One unchanged rule, seven examples, one phone. Every one matched the same
pattern, was repaired by the same code with no architecture-specific branches,
passed host and Android verification, and got faster.

```
Target      RMX2030 · arm64-v8a · Android 10 (SDK 29)
Benchmark   5 warmup + 20 measured iterations, 1 repetition, 4 threads
Backend     ExecuTorch 1.4.0 + XNNPACK
Weights     random (delegation is decided by the graph)
```

Single-run measurements. No median across runs is implied.

## The pattern

`softmax` on a non-last dimension, which the XNNPACK partitioner declines. Six
of these are `segmentation_models_pytorch` architectures configured with the
library's documented `activation="softmax2d"`, which is `nn.Softmax(dim=1)`;
the seventh is a small hand-written classifier. No `forward()` was edited.

## Results

| Architecture | softmax share | Runtime delegation | p50 before | p50 after | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| PSPNet | 46.8% | 53.0% → 99.6% | 224.57 ms | 73.26 ms | **3.07x** |
| DeepLabV3+ | 40.9% | 59.1% → 100.0% | 376.65 ms | 132.67 ms | **2.84x** |
| SmallNet (`interface_mnist`) | 4.5% | 95.4% → 100.0% | 4.63 ms | 2.42 ms | 1.91x |
| Linknet | 41.7% | 58.3% → 100.0% | 258.99 ms | 163.40 ms | 1.58x |
| FPN | 25.2% | 70.8% → 93.4% | 386.79 ms | 253.26 ms | 1.53x |
| U-Net | 22.1% | 73.5% → 93.7% | 525.55 ms | 343.40 ms | 1.53x |
| U-Net++ | 16.2% | 79.5% → 93.4% | 625.03 ms | 424.54 ms | 1.47x |

Every one detected exactly **1 site** and repaired it.

Host correctness PASS, device correctness PASS, backend fidelity OK for all
seven. Every one produced an optimized `.pte`.

## What this shows

Operator-count delegation moves by roughly a point when one operator is
rewritten. Runtime-weighted delegation moved 14–47 points across these seven
models, and p50 latency fell 21–67%.

`interface_mnist` is the useful counter-example: its softmax is only 4.5% of
runtime, yet the repair still returned 1.91x, because removing the fallback
also removed the delegate boundary it was forcing. Hotspot share is a screening
signal, not a prediction - which is why DelegateDoctor benchmarks rather than
estimates.

## Changes required to DD-001

**None.** The existing detection and rewrite ran unchanged on all seven.
