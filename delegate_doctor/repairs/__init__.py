"""Repair rules.

Three accepted rules, applied in order. There is deliberately no registry or
plugin mechanism: at this size a dynamic lookup would add indirection without
adding capability. `ALL_RULES` is a plain list and the pipeline walks it.

Each rule module exposes the same interface:

    detect(exported_program)  -> DetectionResult with .detections / .skipped
    apply(exported_program)   -> int, how many sites were repaired
    describe_rewrite()        -> str
    matches_portable_kernel(kernel_name) -> bool
    RULE_ID, RULE_TITLE

The three rules touch disjoint operators - softmax, alias and avg_pool2d - so
applying all of them to one graph cannot make them interfere.

Entering this list means a rule has evidence that it is correct and, on at least
one supported Arm64 Android target, faster. It does NOT mean the rule is faster
everywhere: `delegate-doctor doctor` still benchmarks original vs repaired on
the user's own device and rejects the repair if it does not win there.
"""

from . import dd001_softmax, dd002_noop_alias, dd003_avgpool_pad

ALL_RULES = [dd001_softmax, dd002_noop_alias, dd003_avgpool_pad]
