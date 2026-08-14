"""DD-003 on a second architecture, in its zero-padding form.

DenseNet's transition layers use avg_pool2d with no padding, where the repair
is a flag change and no node is added at all. DD-003 was never designed against
this model, which is the point.

`weights=None` keeps the first run offline. Note that the DenseNet169 latency
row in the README was measured with pretrained weights; the repair and the
delegation change are identical either way, but the millisecond figures are
not directly reproducible from this file.
"""

import torch
from torchvision.models import densenet169


INPUT_SHAPE = (1, 3, 224, 224)


def delegate_doctor_model():
    """Construct the model DelegateDoctor should analyze."""
    torch.manual_seed(0)

    model = densenet169(
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