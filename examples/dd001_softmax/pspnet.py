import segmentation_models_pytorch as smp
import torch

ENCODER = "mobilenet_v2"
CLASSES = 21
INPUT_SHAPE = (1, 3, 256, 256)
CLASS_DIMENSION = 1


def delegate_doctor_model():
    torch.manual_seed(0)

    model = smp.PSPNet(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=CLASSES,
        activation="softmax2d",
    )

    model.eval()
    return model


def delegate_doctor_inputs():
    torch.manual_seed(0)

    return (
        torch.randn(*INPUT_SHAPE, dtype=torch.float32),
    )