"""Tests for model selection and the concise console output.

Offline: model *construction* is mocked where the real smp import would be slow,
except for a couple of checks that genuinely need the built module. No adb,
device, NDK, network or runner binaries.
"""

import os

import pytest
import torch

from delegate_doctor import models, reporting
from delegate_doctor.cli import BUILTIN_EXAMPLES, load_model_spec


# --- the six model names ---------------------------------------------------

def test_all_six_model_names_are_available():
    assert models.MODEL_NAMES == [
        "unet", "unetplusplus", "fpn", "pspnet", "deeplabv3plus", "linknet",
    ]


def test_every_model_name_has_a_builtin_example_file():
    for name in models.MODEL_NAMES:
        assert name in BUILTIN_EXAMPLES, name
        assert os.path.isfile(BUILTIN_EXAMPLES[name]), BUILTIN_EXAMPLES[name]


def test_every_model_name_maps_to_its_own_architecture(monkeypatch):
    """create_model dispatches to a distinct smp class per name."""
    built = []

    class FakeArchitecture:
        def __init__(self, cls_name):
            self.cls_name = cls_name

        def __call__(self, **kwargs):
            built.append((self.cls_name, kwargs))
            return torch.nn.Identity()

    class FakeSmp:
        Unet = FakeArchitecture("Unet")
        UnetPlusPlus = FakeArchitecture("UnetPlusPlus")
        FPN = FakeArchitecture("FPN")
        PSPNet = FakeArchitecture("PSPNet")
        DeepLabV3Plus = FakeArchitecture("DeepLabV3Plus")
        Linknet = FakeArchitecture("Linknet")

    import sys

    monkeypatch.setitem(sys.modules, "segmentation_models_pytorch", FakeSmp)

    for name in models.MODEL_NAMES:
        models.create_model(name)

    assert [cls for cls, _ in built] == [
        "Unet", "UnetPlusPlus", "FPN", "PSPNet", "DeepLabV3Plus", "Linknet",
    ]


def test_every_model_uses_the_validated_configuration(monkeypatch):
    """mobilenet_v2, 21 classes, softmax2d, no pretrained weights - for all six."""
    captured = []

    class FakeArchitecture:
        def __call__(self, **kwargs):
            captured.append(kwargs)
            return torch.nn.Identity()

    class FakeSmp:
        Unet = UnetPlusPlus = FPN = PSPNet = DeepLabV3Plus = Linknet = FakeArchitecture()

    import sys

    monkeypatch.setitem(sys.modules, "segmentation_models_pytorch", FakeSmp)

    for name in models.MODEL_NAMES:
        models.create_model(name)

    assert len(captured) == len(models.MODEL_NAMES)
    for kwargs in captured:
        assert kwargs["encoder_name"] == "mobilenet_v2"
        assert kwargs["encoder_weights"] is None      # never downloads
        assert kwargs["in_channels"] == 3
        assert kwargs["classes"] == 21
        assert kwargs["activation"] == "softmax2d"    # -> nn.Softmax(dim=1)


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError) as caught:
        models.create_model("resnet")
    assert "Unknown model" in str(caught.value)


def test_unknown_cli_target_lists_the_available_models():
    with pytest.raises(SystemExit) as caught:
        load_model_spec("resnet")
    message = str(caught.value)
    assert "Unknown model: resnet" in message
    for name in models.MODEL_NAMES:
        assert name in message


# --- display metadata ------------------------------------------------------

def test_display_names_are_distinct_and_not_all_unet():
    names = [models.DISPLAY_NAMES[n] for n in models.MODEL_NAMES]
    assert names == ["U-Net", "U-Net++", "FPN", "PSPNet", "DeepLabV3+", "Linknet"]
    assert len(set(names)) == len(names)


def test_model_spec_name_identifies_architecture_and_encoder(monkeypatch):
    monkeypatch.setattr(models, "create_model", lambda name: torch.nn.Identity())
    spec = models.build_model_spec("pspnet")
    assert spec.name == "PSPNet / MobileNetV2"
    assert spec.argmax_dim == 1
    assert tuple(spec.example_inputs[0].shape) == (1, 3, 256, 256)
    assert "softmax2d" in spec.description


def test_input_shape_text():
    assert models.input_shape_text() == "1x3x256x256"


# --- concise console output ------------------------------------------------

class FakeMetrics:
    def __init__(self, max_abs=1.86e-08):
        self.max_absolute_error = max_abs
        self.mean_absolute_error = 3.2e-09
        self.mean_squared_error = 2.0e-17
        self.max_relative_error = 4.2e-07


class FakeVerification:
    def __init__(self, passed=True, reasons=None):
        self.passed = passed
        self.repaired_vs_original = FakeMetrics()
        self.repaired_vs_eager = FakeMetrics()
        self.original_device_vs_host = FakeMetrics()
        self.repaired_device_vs_host = FakeMetrics()
        self.argmax_agreement = 1.0
        self.failure_reasons = reasons or []
        self.error = ""

    @property
    def status_text(self):
        return "PASS" if self.passed else "FAIL"


def test_header_names_the_architecture_and_device():
    text = reporting.format_header(
        "PSPNet / MobileNetV2", "1x3x256x256", "RMX2030 · arm64-v8a · Android 10")
    assert "PSPNet / MobileNetV2" in text
    assert "1x3x256x256" in text
    assert "RMX2030" in text
    assert "U-Net" not in text


def test_successful_verification_is_two_lines_without_metric_dumps():
    text = reporting.format_verification(FakeVerification(), FakeVerification())
    assert "Host:" in text and "Android:" in text
    assert "argmax 100.00%" in text
    # the long per-metric breakdown must not appear on a clean run
    assert "mean abs" not in text
    assert "max rel" not in text
    assert len([line for line in text.splitlines() if line.strip()]) <= 4


def test_failed_verification_prints_full_diagnostics():
    """Conciseness applies to successes; failures must stay debuggable."""
    failing = FakeVerification(passed=False, reasons=["max abs error above tolerance"])
    text = reporting.format_verification(FakeVerification(), failing)
    assert "FAIL" in text
    assert "tolerance:" in text
    assert "mean abs" in text          # full metrics now shown
    assert "max rel" in text
    assert "max abs error above tolerance" in text


def test_detection_output_is_compact():
    class FakeDetection:
        node_name = "softmax"
        input_shape = (1, 21, 256, 256)
        tensor_rank = 4
        softmax_dim = 1
        last_dim = 3
        vector_count = 65536
        vector_length = 21
        element_stride = 65536

    class FakeResult:
        detections = [FakeDetection()]
        skipped = []

    text = reporting.format_detection(FakeResult())
    assert "softmax(dim=1)" in text
    assert "[1, 21, 256, 256]" in text
    assert "65,536 vectors x 21 classes" in text
    # the old multi-line prose explanation is gone
    assert "XNNPACK requires softmax to operate on the" not in text
    assert len([line for line in text.splitlines() if line.strip()]) <= 4


def test_benchmark_output_is_a_compact_table():
    class Stats:
        def __init__(self, p50, p95, mean):
            self.p50_ms, self.p95_ms, self.mean_ms = p50, p95, mean
            self.p99_ms, self.sample_count = p95 + 5, 450
            self.stdev_ms, self.min_ms, self.max_ms = 1.0, p50 - 5, p95 + 9
            self.throughput_per_second = 1000.0 / mean

    class FakeBenchmark:
        before = Stats(242.69, 284.43, 250.31)
        after = Stats(65.53, 73.40, 68.22)
        threads, measured_iterations, repetitions = 4, 150, 3
        device_is_emulator = False
        p50_speedup = 242.69 / 65.53

    text = reporting.format_benchmark(FakeBenchmark())
    assert "p50" in text and "p95" in text and "mean" in text
    assert "3.70x speedup" in text
    assert "73.0% lower p50" in text
    assert len([line for line in text.splitlines() if line.strip()]) <= 9


def test_emulator_benchmark_stays_labelled():
    """Compressing output must not lose the emulator caveat."""
    class Stats:
        p50_ms = p95_ms = mean_ms = 10.0
        p99_ms = 11.0
        sample_count = 450

    class FakeBenchmark:
        before = after = Stats()
        threads, measured_iterations, repetitions = 4, 150, 3
        device_is_emulator = True
        p50_speedup = 1.0

    assert "emulator" in reporting.format_benchmark(FakeBenchmark())


def test_accepted_decision_names_the_device():
    class FakeDecision:
        accepted = True
        headline = "REPAIR ACCEPTED"
        speedup = 3.70
        message = "..."

    text = reporting.format_decision(FakeDecision(), "RMX2030 · arm64-v8a · Android 10")
    assert "ACCEPTED" in text
    assert "3.70x" in text
    assert "RMX2030" in text


def test_rejected_decision_still_explains_why():
    class FakeDecision:
        accepted = False
        headline = "REPAIR REJECTED - Android numerical verification failed"
        speedup = 2.9
        message = "The tensors produced on the Android device did not verify."

    text = reporting.format_decision(FakeDecision(), "RMX2030")
    assert "REJECTED" in text
    assert "did not verify" in text
