import torch

from torchvision.models import (
    ConvNeXt_Small_Weights,
    convnext_small,
)


INPUT_SHAPE = (1, 3, 224, 224)


def delegate_doctor_model():
    weights = ConvNeXt_Small_Weights.DEFAULT

    model = convnext_small(
        weights=weights,
    )

    model.eval()
    return model


def delegate_doctor_inputs():
    torch.manual_seed(0)

    return (
        torch.randn(*INPUT_SHAPE, dtype=torch.float32),
    )