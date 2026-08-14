import segmentation_models_pytorch as smp
import torch


ENCODER = "mobilenet_v2"
CLASSES = 21
INPUT_SHAPE = (1, 3, 256, 256)


def delegate_doctor_model():
    """Construct the model DelegateDoctor should analyze."""
    torch.manual_seed(0)

    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=CLASSES,
        activation="softmax2d",
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