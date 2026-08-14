"""DD-003: nine avg_pool2d fallbacks worth ~60% of measured runtime.

PRETRAINED WEIGHTS, deliberately. The first run downloads a checkpoint from the
PyTorch hub.

Unlike the other examples this one keeps its trained weights, because the
recorded DD-003 evidence was measured with them and the numbers in the README
would not be reproducible otherwise. A randomly initialised Inception also
produces logits around 1e10, where fp32 comparisons stop meaning anything.
"""

import torch
from torchvision.models import Inception_V3_Weights, inception_v3

INPUT_SHAPE = (1, 3, 299, 299)


def delegate_doctor_model():
    # `aux_logits` is deliberately not passed. TorchVision forces it to True
    # whenever pretrained weights are requested - the checkpoint contains the
    # auxiliary head - and passing False raises during construction. In eval
    # mode the auxiliary output is dropped anyway, so the exported graph is
    # the deployment graph either way.
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.eval()
    return model


def delegate_doctor_inputs():
    torch.manual_seed(0)
    return (torch.randn(*INPUT_SHAPE, dtype=torch.float32),)