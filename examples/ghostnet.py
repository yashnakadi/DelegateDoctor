"""Demo workload: timm GhostNet-100, an ImageNet classifier.

Architecture:  GhostNet-100 (from timm)
Task:          ImageNet classification, 1000 classes
Input:         1 x 3 x 224 x 224
Weights:       pretrained=False

This model is NOT part of DelegateDoctor. It is the DD-002 demonstration, and it
deliberately comes from a different codebase and a different task than the six
segmentation examples.

The fallback is not planted. timm's `GhostModule.forward` ends with
`out[:, :self.out_chs, :, :]`; when that slice covers the whole tensor it
exports as `aten.alias`, a pure no-op. XNNPACK has no config for alias, so all
32 of them drop out of the delegate and split the graph into 49 blobs.

Run:  delegate-doctor doctor ghostnet
"""

from delegate_doctor.export_model import ModelSpec
from delegate_doctor.models import build_model_spec, create_ghostnet


def create_model():
    """The bare GhostNet, without DelegateDoctor's wrapper."""
    return create_ghostnet()


def build_model() -> ModelSpec:
    """What `delegate-doctor doctor ghostnet` loads."""
    return build_model_spec("ghostnet")
