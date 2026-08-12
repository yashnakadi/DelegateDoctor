"""U-Net with a MobileNetV2 encoder, 21 classes - the original DD-001 demonstration.

    python examples/unet.py

Nothing here is DelegateDoctor-specific. This file builds a stock
`segmentation_models_pytorch` U-Net and hands it to the same public
`optimize()` that any user calls:

    from delegate_doctor import optimize
    optimize(model, args=(example_input,))

DelegateDoctor has no idea what a U-Net is. It exports whatever model it is
given and analyzes the resulting graph.

Why this model is interesting
-----------------------------
The fallback DD-001 repairs is not planted here. `activation="softmax2d"` is a
documented smp constructor argument, and the library implements it as
`nn.Softmax(dim=1)` on the full-resolution (1, 21, 256, 256) class tensor.
XNNPACK only delegates a last-dimension softmax, so the pattern appears through
this architecture's own decoder.

`encoder_weights=None` keeps the example offline and deterministic. Graph
structure and latency do not depend on the weight values, and no claim is made
about segmentation accuracy.

This is the configuration the recorded DD-001 evidence was measured with; see
results/dd001_segmentation_generalization.md.
"""

import segmentation_models_pytorch as smp
import torch

from delegate_doctor import optimize

ENCODER = "mobilenet_v2"
CLASSES = 21                 # Pascal VOC
INPUT_SHAPE = (1, 3, 256, 256)

# Segmentation output is (batch, classes, height, width), so the class dimension
# is 1. Verification uses it to check every pixel keeps its predicted class.
CLASS_DIMENSION = 1


def build_model():
    """A stock smp U-Net, built exactly as the recorded evidence was."""
    torch.manual_seed(0)
    return smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=CLASSES,
        activation="softmax2d",     # -> nn.Softmax(dim=1) in smp's head
    )


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
