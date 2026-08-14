import torch


class SmallNet(torch.nn.Module):
    def __init__(self, classes: int = 10):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Conv2d(32, classes, kernel_size=1)

    def forward(self, x):
        x = self.head(self.features(x))
        # Over the channel dimension, which is not the last one. This is the
        # pattern DD-001 recognises.
        return torch.softmax(x, dim=1)


def delegate_doctor_model():
    model = SmallNet()
    model.eval()
    return model


def delegate_doctor_inputs():
    torch.manual_seed(0)
    return (
        torch.randn(1, 1, 28, 28, dtype=torch.float32),
    )
