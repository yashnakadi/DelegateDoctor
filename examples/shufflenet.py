import torch
from torchvision.models import shufflenet_v2_x1_0


def create_model():
    model = shufflenet_v2_x1_0(weights=None)
    model.eval()
    return model


def example_inputs():
    input_tensor = torch.randn(1, 3, 224, 224)
    return (input_tensor,)