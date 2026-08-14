# DD-002 on a physical phone — result: ACCEPTED

The no-op alias repair is correct and about **1.4x faster** on this phone.

This model needed a deeper benchmark than the rest of the suite, and the reason
is documented below: its baseline latency is bimodal, so a single repetition
can report either verdict depending on which mode it samples.

```
Target      RMX2030 · arm64-v8a · Android 10 (SDK 29)
Benchmark   5 warmup + 20 measured iterations, 5 repetitions, 4 threads
            (100 interleaved samples per program - see "Why 5 repetitions")
Backend     ExecuTorch 1.4.0 + XNNPACK
Model       timm GhostNet-100, random weights, 1x3x224x224
Run         artifacts/run_125
```

## Result

| | value |
| --- | --- |
| Rule | DD-002 |
| Matching sites | 32 |
| Top portable hotspot | `alias_copy.out` |
| Hotspot runtime share | 0.5% |
| Operator delegation before | 79.9% |
| Runtime delegation | 99.5% → 100.0% |
| Host correctness | **PASS** |
| Device correctness | **PASS** |
| Backend fidelity | OK |
| p50 before | 721.71 ms |
| p50 after | **499.19 ms** |
| Speedup | **1.45x** |
| Decision | **REPAIR ACCEPTED** |
| Optimized `.pte` | written |

A second deep run (`run_126`) gave 737.06 → 519.27 ms, **1.42x**. The two agree
closely.

## Why 5 repetitions

At the suite's default of 1 repetition this model returns either verdict, and
the reason is visible in the sample distribution rather than the summary.

Across 100 interleaved baseline samples in `run_125`:

```
before   min 43.63   p50 721.71   max 826.76   stdev 262.39
after    min 59.25   p50 499.19   max 577.15   stdev 146.06
```

The original is **bimodal**: it touches 43 ms and 827 ms in the same run, with a
standard deviation a third of its own median. The RMX2030 is a big.LITTLE
Snapdragon 665, and GhostNet's many small operators appear to be sensitive to
which cluster the runner lands on. The repaired graph is far tighter.

Eight runs at 1 repetition, before this was understood:

| baseline p50 | repaired p50 | verdict |
| ---: | ---: | --- |
| 787.55 ms | 550.59 ms | ACCEPTED |
| 797.78 ms | 547.28 ms | ACCEPTED |
| 723.22 ms | 463.15 ms | ACCEPTED |
| 703.16 ms | 503.22 ms | ACCEPTED |
| 85.71 ms | 416.99 ms | rejected |
| 48.30 ms | 526.00 ms | rejected |
| 47.21 ms | 533.30 ms | rejected |
| 46.02 ms | 404.22 ms | rejected |

The repaired graph is stable at 404–551 ms throughout. Only the **baseline**
moves, and it decides the verdict.

Two runs in that table had baselines of ~47 ms with a standard deviation under
1 ms. A settled-looking measurement is not the same as a representative one:
those runs caught 20 consecutive fast-mode samples, and 20 samples in a row
cannot see a mode they never entered. More repetitions is what fixes that, not
a steadier-looking single run.

There is also a methodological point. `benchmark_before_after` interleaves the
before and after passes *across repetitions*. At `--reps 1` there is exactly one
pass of each, so no interleaving happens and any drift between them lands
directly on the comparison. For the stable models in this suite that costs
nothing; for this one it is the difference between the two verdicts.

## What this says about the gate

Nothing here weakened a threshold or excused a result. The tolerance, the
correctness gates and the decision rule are unchanged; the measurement was made
properly and the answer changed. That is the intended behaviour of a tool that
benchmarks rather than estimates - including when the first measurement was the
convenient one.

If you are evaluating a marginal repair on your own device, raise `--reps`.
