"""Demo workload: Linknet with a MobileNetV2 encoder, 21 classes.

Architecture:  Linknet (residual encoder-decoder)
Encoder:       mobilenet_v2 (encoder_weights=None)
Classes:       21 (Pascal VOC)
Input:         1 x 3 x 256 x 256
Activation:    softmax2d  ->  nn.Softmax(dim=1) inside smp's head

This model is NOT part of DelegateDoctor. It is a real architecture to point the
tool at, which is why it lives outside the package.

The fallback DD-001 repairs is not planted here. `activation="softmax2d"` is a
documented `segmentation_models_pytorch` constructor argument, and the library
implements it as `nn.Softmax(dim=1)` on the full-resolution
(1, 21, 256, 256) class tensor. XNNPACK only delegates a last-dimension softmax,
so the pattern appears naturally.

Run:  delegate-doctor doctor linknet
"""

from delegate_doctor.export_model import ModelSpec
from delegate_doctor.models import build_model_spec, create_linknet


def create_model():
    """The bare Linknet, if you want it without DelegateDoctor's wrapper."""
    return create_linknet()


def build_model() -> ModelSpec:
    """What `delegate-doctor doctor linknet` loads."""
    return build_model_spec("linknet")
