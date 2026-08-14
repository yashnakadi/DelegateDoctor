import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


INPUT_SHAPE = (1, 3, 224, 224)


def delegate_doctor_model():
    """Construct the model DelegateDoctor should analyze."""
    model = mobilenet_v2(
        weights=MobileNet_V2_Weights.DEFAULT,
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