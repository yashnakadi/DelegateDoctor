# DD-003 on a physical phone — result: ACCEPTED

`avg_pool2d` with `count_include_pad=True` is rejected outright by the XNNPACK
partitioner. DD-003 materializes the padding so the pooling operator has none
left to argue about. On this phone both matching models got faster and stayed
correct.

```
Target      RMX2030 · arm64-v8a · Android 10 (SDK 29)
Benchmark   5 warmup + 20 measured iterations, 1 repetition, 4 threads
Backend     ExecuTorch 1.4.0 + XNNPACK
Weights     random (delegation is decided by the graph)
```

Single-run measurements. No median across runs is implied.

## Results

| Model | Sites | `avg_pool2d` share | Runtime delegation | p50 before | p50 after | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Inception V3 | 9 | **46.7%** | 53.3% → 100.0% | 825.18 ms | 424.19 ms | **1.95x** |
| DenseNet169 | 3 | 4.4% | 95.5% → 99.9% | 343.51 ms | 323.01 ms | 1.06x |

Host correctness PASS, device correctness PASS, backend fidelity OK for both.
Both produced an optimized `.pte`.

## Why Inception V3 is the headline

| Inception V3, before repair | |
| --- | --- |
| Operators delegated | **97.1%** (9 of 314 fell back) |
| Runtime inside XNNPACK | **53.3%** |

Nine `avg_pool2d` nodes - under 3% of the graph - carried 46.7% of measured
inference time. A tool reporting operator counts would have called this model
well optimized and moved on.

After the repair the model is 100% runtime-delegated and 1.95x faster.

## DenseNet169 matters for a different reason

DD-003 was never written against DenseNet. Its transition layers hit the
*zero-padding* form of the same rule, where the repair is a flag change and no
node is added at all. The rule matched a model it had never seen, which is the
difference between a graph pattern and a hard-coded special case.

Its 1.06x is small because its pooling was only 4.4% of runtime - which is
exactly what runtime-weighted ranking predicts, and exactly why DelegateDoctor
measures instead of assuming.

## Equivalence

The transformation is bit-exact in eager PyTorch (0.000e+00 across a sweep of
shapes, kernels, strides and paddings), and the partitioner's own rejection
reason is quoted from ExecuTorch source. That analysis is in
[`dd003_avgpool_research.md`](dd003_avgpool_research.md); its timing tables
have been superseded by this file.
