"""Optional AI assistance, kept strictly outside the deterministic pipeline.

    privacy.py      what may leave this machine, and what is scrubbed first
    credentials.py  where the API key lives, and how it never escapes

The rule the whole package exists to enforce:

    The agent proposes. DelegateDoctor verifies.

Nothing here decides whether a model is supported, whether a rewrite is correct,
or whether an optimization is faster. `torch.export`, ExecuTorch, the host and
device verification gates and the target benchmark decide those, exactly as they
did before any of this existed.
"""
