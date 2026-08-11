"""The segmentation architectures DelegateDoctor ships as demo workloads.

All six are stock `segmentation_models_pytorch` architectures built with one
shared configuration. They are here, rather than duplicated in six example
files, so that `cli.py` and `examples/` share exactly one implementation.

Why these six
-------------
Every one of them is configured with `activation="softmax2d"`, which smp
implements as `nn.Softmax(dim=1)` in its own segmentation head. That is the
documented way to make a multi-class segmentation model emit probabilities, and
it produces the DD-001 pattern naturally in all six:

    softmax(dim=1) on [1, 21, 256, 256]   (rank 4, softmax dim 1, last dim 3)

Nothing is inserted or patched to create the fallback. Each architecture reaches
it through its own decoder, which is what makes them useful evidence that DD-001
is a model-independent repair rather than a U-Net workaround.

`encoder_weights=None` keeps the examples offline and deterministic. Latency and
graph structure do not depend on the weight values, and no claim is made about
segmentation accuracy.
"""

from __future__ import annotations

import torch

# One controlled configuration, shared by every architecture.
ENCODER = "mobilenet_v2"
ENCODER_DISPLAY = "MobileNetV2"
CLASSES = 21                 # Pascal VOC
IN_CHANNELS = 3
INPUT_HEIGHT = 256
INPUT_WIDTH = 256
ACTIVATION = "softmax2d"     # -> nn.Softmax(dim=1) inside smp's head

# Segmentation output is (batch, classes, height, width), so the class
# dimension is 1. Verification uses it to check that every pixel keeps its
# predicted class.
CLASS_DIMENSION = 1


def _build(architecture_class, seed: int = 0):
    """Instantiate one smp architecture with the shared configuration."""
    torch.manual_seed(seed)
    return architecture_class(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=CLASSES,
        activation=ACTIVATION,
    ).eval()


def create_unet():
    import segmentation_models_pytorch as smp

    return _build(smp.Unet)


def create_unetplusplus():
    import segmentation_models_pytorch as smp

    return _build(smp.UnetPlusPlus)


def create_fpn():
    import segmentation_models_pytorch as smp

    return _build(smp.FPN)


def create_pspnet():
    import segmentation_models_pytorch as smp

    return _build(smp.PSPNet)


def create_deeplabv3plus():
    import segmentation_models_pytorch as smp

    return _build(smp.DeepLabV3Plus)


def create_linknet():
    import segmentation_models_pytorch as smp

    return _build(smp.Linknet)


# Display names, so the report never calls a PSPNet a U-Net.
DISPLAY_NAMES = {
    "unet": "U-Net",
    "unetplusplus": "U-Net++",
    "fpn": "FPN",
    "pspnet": "PSPNet",
    "deeplabv3plus": "DeepLabV3+",
    "linknet": "Linknet",
}

MODEL_NAMES = list(DISPLAY_NAMES)


def create_model(name: str):
    """Build one architecture by its CLI name."""
    if name == "unet":
        return create_unet()
    if name == "unetplusplus":
        return create_unetplusplus()
    if name == "fpn":
        return create_fpn()
    if name == "pspnet":
        return create_pspnet()
    if name == "deeplabv3plus":
        return create_deeplabv3plus()
    if name == "linknet":
        return create_linknet()
    raise ValueError(f"Unknown model: {name}")


def example_inputs():
    """The single deterministic input shape every example uses."""
    return (torch.randn(1, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH),)


def input_shape_text() -> str:
    return f"1x{IN_CHANNELS}x{INPUT_HEIGHT}x{INPUT_WIDTH}"


def build_model_spec(name: str):
    """Build the ModelSpec DelegateDoctor's pipeline consumes."""
    from .export_model import ModelSpec

    return ModelSpec(
        name=f"{DISPLAY_NAMES[name]} / {ENCODER_DISPLAY}",
        model=create_model(name),
        example_inputs=example_inputs(),
        argmax_dim=CLASS_DIMENSION,
        description=(
            f"segmentation_models_pytorch {DISPLAY_NAMES[name]}, "
            f"{ENCODER} encoder, {CLASSES} classes, "
            f"{INPUT_HEIGHT}x{INPUT_WIDTH} input, activation={ACTIVATION}"
        ),
    )
