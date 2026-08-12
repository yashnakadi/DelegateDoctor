"""timm GhostNet-100, an ImageNet classifier - the DD-002 demonstration.

    python examples/ghostnet.py

Nothing here is DelegateDoctor-specific. This file builds a stock timm
GhostNet-100 and hands it to the same public `optimize()` that any user calls:

    from delegate_doctor import optimize
    optimize(model, args=(example_input,))

Deliberately a different codebase and a different task from the six
segmentation examples: the same unchanged DelegateDoctor analyzes both, because
it has no architecture-specific code for either.

Why this model is interesting
-----------------------------
The fallback is not planted. timm's `GhostModule.forward` ends with
`out[:, :self.out_chs, :, :]`; when that slice covers the whole tensor it
exports as `aten.alias`, a pure no-op. XNNPACK has no config for alias, so all
32 of them drop out of the delegate and split the graph into 49 blobs - which is
exactly the pattern DD-002 removes.

`pretrained=False` keeps the example offline and deterministic. Graph structure
and latency do not depend on the weight values, and no claim is made about
classification accuracy.

This is the configuration the recorded DD-002 evidence was measured with; see
results/dd002_emulator_validation.md.
"""

import timm
import torch

from delegate_doctor import optimize

MODEL = "ghostnet_100"
INPUT_SHAPE = (1, 3, 224, 224)

# ImageNet logits are (batch, 1000), so an argmax over dim 1 is top-1 - the
# meaningful semantic check for a classifier.
CLASS_DIMENSION = 1


def build_model():
    """A stock timm GhostNet-100, built exactly as the recorded evidence was."""
    torch.manual_seed(0)
    return timm.create_model(MODEL, pretrained=False)


if __name__ == "__main__":
    model = build_model()
    model.eval()

    example_input = torch.randn(*INPUT_SHAPE, dtype=torch.float32)

    result = optimize(
        model,
        args=(example_input,),
        argmax_dim=CLASS_DIMENSION,
    )

    # Opens the self-contained HTML report in the developer's browser. The
    # analysis is already complete and saved; this only displays it.
    result.open_report()
