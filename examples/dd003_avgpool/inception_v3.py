"""DD-003: avg_pool2d(count_include_pad=True), which XNNPACK rejects outright.

TorchVision's Inception V3 calls `F.avg_pool2d(x, kernel_size=3, stride=1,
padding=1)` in every BasicConv2d branch - nine times - and never mentions
`count_include_pad`, so it takes the ATen default of True. The XNNPACK
partitioner refuses that unconditionally, so all nine fall back to the portable
kernel.

    delegate-doctor optimize examples/dd003_avgpool/inception_v3.py

Random weights: delegation is decided by the graph, and the first run needs no
network.

`init_weights=False` is deliberate and load-bearing. TorchVision's default
(True) applies a legacy truncated-normal initialisation whose activations grow
through 300 unnormalised operators until the logits reach ~1e12 - a scale at
which the 1e-5 absolute correctness tolerance means nothing, and every
comparison fails on rounding. False leaves PyTorch's own layer defaults, which
stay bounded (~0.02), so the phone's verdict is about the repair rather than
about the initialisation.
"""

import torch
from torchvision.models import inception_v3

INPUT_SHAPE = (1, 3, 299, 299)


def delegate_doctor_model():
    # `aux_logits` is deliberately not passed. In eval mode the auxiliary
    # output is dropped anyway, so the exported graph is the deployment graph.
    torch.manual_seed(0)
    model = inception_v3(weights=None, init_weights=False)
    model.eval()
    return model


def delegate_doctor_inputs():
    torch.manual_seed(0)
    return (torch.randn(*INPUT_SHAPE, dtype=torch.float32),)
