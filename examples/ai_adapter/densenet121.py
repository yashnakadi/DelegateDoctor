import torch
import torch.nn as nn
from torchvision.models import densenet121


INPUT_SHAPE = (1, 3, 224, 224)


class DenseNetModel(nn.Module):
    def __init__(self):
        super().__init__()

        torch.manual_seed(0)

        self.model = densenet121(
            weights=None,
        )

    def forward(self, x):
        return self.model(x)