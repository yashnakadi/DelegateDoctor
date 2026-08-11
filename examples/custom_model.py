"""Example of the `delegate-doctor optimize` contract.

This is what a user's own model file looks like. DelegateDoctor treats it
exactly like any external file - nothing detects that it lives in examples/.

    delegate-doctor optimize examples/custom_model.py

A model file defines two functions and nothing else is required:

    create_model()    -> torch.nn.Module   (you load your own weights here)
    example_inputs()  -> tuple of positional fp32 tensors

This example uses a stock segmentation_models_pytorch FPN, which naturally
contains the DD-001 pattern (its softmax2d head normalises over the class
dimension). Nothing was added to make a repair apply.
"""

import torch
import segmentation_models_pytorch as smp


def create_model():
    """Build the model. Load your own weights here if you have them:

        state = torch.load("weights.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    """
    model = smp.FPN(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=3,
        classes=21,
        activation="softmax2d",
    )
    model.eval()
    return model


def example_inputs():
    """Positional inputs for the model's forward(). Called once per run."""
    torch.manual_seed(0)
    return (torch.randn(1, 3, 256, 256),)
