"""Portable fallbacks that no DelegateDoctor rule recognises.

An honest negative result: DelegateDoctor ranks the fallbacks by measured
runtime and reports that nothing in the catalog matches them. The analysis is
still the product.

`weights=None` on purpose. Which operators XNNPACK accepts is a property of the
graph, so random weights show the same fallbacks as trained ones, and the first
run needs no network.
"""

import torch

from torchvision.models import convnext_small


INPUT_SHAPE = (1, 3, 224, 224)


def delegate_doctor_model():
    model = convnext_small(
        weights=None,
    )

    model.eval()
    return model


def delegate_doctor_inputs():
    torch.manual_seed(0)

    return (
        torch.randn(*INPUT_SHAPE, dtype=torch.float32),
    )
