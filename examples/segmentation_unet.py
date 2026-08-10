"""Demonstration workload: U-Net with a MobileNetV2 encoder, 21 classes.

This model is NOT part of DelegateDoctor. It is a realistic workload to point
the tool at, and it lives outside the package so the tool itself never depends
on any model library.

Why this model
--------------
`segmentation_models_pytorch` is a widely used segmentation library. Its
segmentation head takes an `activation` argument, and `activation="softmax2d"`
is the documented way to make a multi-class model output probability maps. The
library implements that as `nn.Softmax(dim=1)` in
`segmentation_models_pytorch/base/modules.py`.

So the non-last-dimension softmax that DD-001 repairs is the library's own
code, applied to a full-resolution (1, 21, 256, 256) tensor. Nothing was
inserted or modified to create the fallback - it is simply how multi-class
segmentation models are normally written.

`encoder_weights=None` keeps the example offline and deterministic. Graph
structure and latency do not depend on the weight values, and this example
makes no claim about segmentation accuracy.
"""

import torch

from delegate_doctor.export_model import ModelSpec

INPUT_HEIGHT = 256
INPUT_WIDTH = 256
NUM_CLASSES = 21                # Pascal VOC
ENCODER_NAME = "mobilenet_v2"

# Segmentation output is (batch, classes, height, width), so the class
# dimension is 1. Verification uses this to check that every pixel keeps its
# predicted class.
CLASS_DIMENSION = 1


def build_model() -> ModelSpec:
    """Build the U-Net and describe it to DelegateDoctor."""
    import segmentation_models_pytorch as smp

    torch.manual_seed(0)
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation="softmax2d",   # -> nn.Softmax(dim=1) inside smp's head
    ).eval()

    example_inputs = (torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH),)

    return ModelSpec(
        name="U-Net / MobileNetV2",
        model=model,
        example_inputs=example_inputs,
        argmax_dim=CLASS_DIMENSION,
        description=(
            f"segmentation_models_pytorch U-Net, {NUM_CLASSES} classes, "
            f"{INPUT_HEIGHT}x{INPUT_WIDTH} input"
        ),
    )
