import torch
import torchvision.models as models


INPUT_SHAPE = (1, 3, 224, 224)


def delegate_doctor_model():
    torch.manual_seed(0)

    model = models.resnext50_32x4d(
        weights=None,
    )

    model.eval()
    return model


def delegate_doctor_inputs():
    torch.manual_seed(0)

    return (
        torch.randn(*INPUT_SHAPE, dtype=torch.float32),
    )