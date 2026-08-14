"""A model that is already fully delegated to XNNPACK.

The useful output here is the analysis, not a repair: DelegateDoctor reports
high runtime delegation and no repair required. That is a real answer.

`weights=None` on purpose. Delegation is decided by the graph, so random
weights demonstrate exactly the same thing as trained ones - and the first run
needs no network.
"""

import torch
from torchvision.models import mobilenet_v2


INPUT_SHAPE = (1, 3, 224, 224)


def delegate_doctor_model():
    """Construct the model DelegateDoctor should analyze."""
    torch.manual_seed(0)

    model = mobilenet_v2(
        weights=None,
    )

    model.eval()
    return model


def delegate_doctor_inputs():
    """Return representative inputs for torch.export."""
    torch.manual_seed(0)

    example_input = torch.randn(
        *INPUT_SHAPE,
        dtype=torch.float32,
    )

    return (example_input,)
