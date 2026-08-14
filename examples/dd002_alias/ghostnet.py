import timm
import torch


MODEL = "ghostnet_100"
INPUT_SHAPE = (1, 3, 224, 224)


def delegate_doctor_model():
    """Construct the model DelegateDoctor should analyze."""
    torch.manual_seed(0)

    model = timm.create_model(
        MODEL,
        pretrained=False,
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