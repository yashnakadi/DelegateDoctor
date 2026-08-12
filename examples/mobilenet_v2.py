"""TorchVision MobileNetV2 - the healthy baseline.

    python examples/mobilenet_v2.py

The other examples all contain a known fallback. This one is here to show what
DelegateDoctor says when there is nothing wrong: MobileNetV2 is a mainstream,
mobile-targeted architecture that XNNPACK is happy to take, so the expected
result is

    FULLY_DELEGATED   or   NO_REPAIR_REQUIRED

and no repaired artifact. DelegateDoctor does not invent an optimization for a
model that does not need one, and reporting that plainly is the point.

NOTE: unlike every other example, this one uses **pretrained weights**, so the
first run downloads them from TorchVision into your torch hub cache. That is
this script's choice, not DelegateDoctor's - the tool never downloads anything.
Pass `weights=None` below if you would rather stay offline; the graph structure,
and therefore the analysis, is identical either way.
"""

import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from delegate_doctor import optimize

INPUT_SHAPE = (1, 3, 224, 224)

# ImageNet logits are (batch, 1000), so an argmax over dim 1 is top-1.
CLASS_DIMENSION = 1


def build_model():
    """Stock TorchVision MobileNetV2 with its published ImageNet weights."""
    return mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)


if __name__ == "__main__":
    model = build_model()
    model.eval()

    torch.manual_seed(0)
    example_input = torch.randn(*INPUT_SHAPE, dtype=torch.float32)

    result = optimize(
        model,
        args=(example_input,),
        argmax_dim=CLASS_DIMENSION,
    )

    # Opens the self-contained HTML report in the developer's browser. The
    # analysis is already complete and saved; this only displays it.
    result.open_report()
