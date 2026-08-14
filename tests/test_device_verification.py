"""Tests for device-side numerical verification.

Fully offline: no adb, no emulator, no NDK, no network, no native build. The
adb boundary is mocked, and tensor files are written to a tmp_path.
"""

import numpy
import pytest
import torch

from delegate_doctor import device, device_verification
from delegate_doctor.device_verification import (
    DeviceVerificationError,
    TensorSpec,
    capture_device_output,
    load_device_tensor,
    run_device_verification,
    spec_from_host_tensor,
    verify_device_outputs,
)


def make_output(seed: int = 0, shape=(1, 4, 8, 8)) -> torch.Tensor:
    """A small stand-in for a segmentation probability map."""
    torch.manual_seed(seed)
    return torch.softmax(torch.randn(*shape), dim=1)


def write_device_file(path, tensor: torch.Tensor):
    """Write raw float32 bytes the way the Android runner does."""
    tensor.detach().numpy().astype(numpy.float32).tofile(str(path))
    return str(path)


# --- binary parsing --------------------------------------------------------

def test_valid_float32_tensor_round_trips(tmp_path):
    original = make_output()
    path = write_device_file(tmp_path / "out.bin", original)

    loaded = load_device_tensor(path, spec_from_host_tensor(original))

    assert loaded.shape == original.shape
    assert loaded.dtype == torch.float32
    assert torch.equal(loaded, original)


def test_missing_file_is_rejected(tmp_path):
    spec = spec_from_host_tensor(make_output())
    with pytest.raises(DeviceVerificationError) as caught:
        load_device_tensor(str(tmp_path / "absent.bin"), spec)
    assert "did not produce an output tensor" in str(caught.value)


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "out.bin"
    path.write_bytes(b"")
    with pytest.raises(DeviceVerificationError) as caught:
        load_device_tensor(str(path), spec_from_host_tensor(make_output()))
    assert "empty" in str(caught.value)


def test_truncated_data_is_rejected(tmp_path):
    original = make_output()
    path = tmp_path / "out.bin"
    truncated = original.flatten()[: original.numel() // 2]
    write_device_file(path, truncated)

    with pytest.raises(DeviceVerificationError) as caught:
        load_device_tensor(str(path), spec_from_host_tensor(original))
    assert "wrong size" in str(caught.value)


def test_extra_unexpected_bytes_are_rejected(tmp_path):
    original = make_output()
    path = tmp_path / "out.bin"
    write_device_file(path, original)
    with open(path, "ab") as handle:
        handle.write(b"\x00\x00\x00\x00")

    with pytest.raises(DeviceVerificationError) as caught:
        load_device_tensor(str(path), spec_from_host_tensor(original))
    assert "wrong size" in str(caught.value)


def test_a_differently_shaped_tensor_of_the_same_size_is_not_silently_reshaped(tmp_path):
    """Byte count is validated against a known shape, never used to guess one."""
    original = make_output(shape=(1, 4, 8, 8))
    other_shape = make_output(shape=(1, 8, 4, 8))
    assert original.numel() == other_shape.numel()

    path = write_device_file(tmp_path / "out.bin", other_shape)
    loaded = load_device_tensor(path, spec_from_host_tensor(original))

    # Same byte count, so it loads - but into the shape we asked for, and the
    # values then disagree, which the comparison step is what catches.
    assert loaded.shape == original.shape
    assert not torch.equal(loaded, original)


def test_unsupported_dtype_is_rejected():
    with pytest.raises(DeviceVerificationError) as caught:
        spec_from_host_tensor(torch.zeros(2, 2, dtype=torch.float64))
    assert "float32" in str(caught.value)


def test_spec_reports_expected_bytes():
    spec = spec_from_host_tensor(make_output(shape=(1, 21, 4, 4)))
    assert spec.element_count == 21 * 16
    assert spec.expected_bytes == 21 * 16 * 4
    assert "dtype=float32" in spec.describe()
    assert "shape=1,21,4,4" in spec.describe()


# --- device command construction -------------------------------------------

class FakeAdb:
    """Records what the device module was asked to do."""

    def __init__(self, tmp_path, produce_output=True, tensor=None):
        self.tmp_path = tmp_path
        self.produce_output = produce_output
        self.tensor = tensor
        self.commands = []
        self.pushed = []
        self.pulled = []
        self.removed = []
        self.serials = []

    def install(self, monkeypatch):
        monkeypatch.setattr(device, "prepare_work_dir",
                            lambda serial=None: self.serials.append(serial))
        monkeypatch.setattr(device, "push_runner",
                            lambda path, serial=None: self.serials.append(serial))

        def push_file(local, remote_name=None, serial=None):
            self.pushed.append((local, remote_name))
            self.serials.append(serial)
            return f"{device.DEVICE_WORK_DIR}/{remote_name}"

        def run_on_device(command, serial=None):
            self.commands.append(command)
            self.serials.append(serial)

        def pull_file(remote, local, serial=None):
            self.pulled.append((remote, local))
            self.serials.append(serial)
            if not self.produce_output:
                raise RuntimeError("nothing to pull")
            write_device_file(local, self.tensor)
            return local

        def remove_remote_files(pattern, serial=None):
            self.removed.append(pattern)

        monkeypatch.setattr(device, "push_file", push_file)
        monkeypatch.setattr(device, "run_on_device", run_on_device)
        monkeypatch.setattr(device, "pull_file", pull_file)
        monkeypatch.setattr(device, "remove_remote_files", remove_remote_files)


def test_capture_uses_the_selected_serial_and_unique_names(tmp_path, monkeypatch):
    tensor = make_output()
    fake = FakeAdb(tmp_path, tensor=tensor)
    fake.install(monkeypatch)

    before = capture_device_output(
        pte_path="before.pte", input_paths=["input.bin"],
        bench_runner_path="runners/executor_runner_bench",
        label="before", output_dir=str(tmp_path), serial="emulator-5554",
    )
    after = capture_device_output(
        pte_path="after.pte", input_paths=["input.bin"],
        bench_runner_path="runners/executor_runner_bench",
        label="after", output_dir=str(tmp_path), serial="emulator-5554",
    )

    # every adb interaction targeted the selected device
    assert set(fake.serials) == {"emulator-5554"}

    # remote model, input and output names differ between the two runs
    remote_names = [remote for _, remote in fake.pushed]
    assert "before_verify_model.pte" in remote_names
    assert "after_verify_model.pte" in remote_names
    assert fake.commands[0] != fake.commands[1]
    assert "--output_file before_output" in fake.commands[0]
    assert "--output_file after_output" in fake.commands[1]

    # local files are distinct too, so one cannot overwrite the other
    assert before != after
    assert before.endswith("before_device_output.bin")
    assert after.endswith("after_device_output.bin")

    # the pulled path is the runner's "<name>-<index>.bin" convention
    assert fake.pulled[0][0].endswith("before_output-0.bin")


def test_capture_pushes_the_same_input_for_both_models(tmp_path, monkeypatch):
    fake = FakeAdb(tmp_path, tensor=make_output())
    fake.install(monkeypatch)

    for label in ("before", "after"):
        capture_device_output(
            pte_path=f"{label}.pte", input_paths=["shared_input.bin"],
            bench_runner_path="bench", label=label,
            output_dir=str(tmp_path), serial="emulator-5554",
        )

    pushed_inputs = [local for local, _ in fake.pushed if local == "shared_input.bin"]
    assert len(pushed_inputs) == 2, "both runs must use the identical input file"


def test_capture_runs_exactly_one_iteration_and_prints_no_tensors(tmp_path, monkeypatch):
    """Verification is a single untimed run; it must not print tensor text."""
    fake = FakeAdb(tmp_path, tensor=make_output())
    fake.install(monkeypatch)

    capture_device_output(
        pte_path="before.pte", input_paths=["input.bin"], bench_runner_path="bench",
        label="before", output_dir=str(tmp_path), serial="emulator-5554",
    )

    command = fake.commands[0]
    assert "--num_executions 1" in command
    assert "--print_output none" in command
    assert device.BENCH_RUNNER_NAME in command


def test_missing_device_output_is_reported_with_both_paths(tmp_path, monkeypatch):
    fake = FakeAdb(tmp_path, produce_output=False, tensor=make_output())
    fake.install(monkeypatch)

    with pytest.raises(DeviceVerificationError) as caught:
        capture_device_output(
            pte_path="before.pte", input_paths=["input.bin"], bench_runner_path="bench",
            label="before", output_dir=str(tmp_path), serial="emulator-5554",
        )
    message = str(caught.value)
    assert "before" in message
    assert "expected on device" in message
    assert "expected locally" in message


def test_benchmark_still_writes_no_tensors():
    """Guard: the timed benchmark path must never gain an --output_file."""
    import inspect

    from delegate_doctor import benchmarking

    source = inspect.getsource(benchmarking.run_one_pass)
    assert "--print_output none" in source
    assert "--output_file" not in source


# --- verification ----------------------------------------------------------

def verify(device_original, device_repaired, host_original, host_repaired, dim=1):
    return verify_device_outputs(
        original_device_output=device_original,
        repaired_device_output=device_repaired,
        original_host_output=host_original,
        repaired_host_output=host_repaired,
        argmax_dim=dim,
    )


def test_identical_device_outputs_pass():
    host = make_output()
    result = verify(host.clone(), host.clone(), host, host)
    assert result.passed
    assert result.status_text == "PASS"
    assert result.argmax_agreement == 1.0


def test_tiny_floating_point_differences_pass():
    host = make_output()
    device_repaired = host + 2e-8
    result = verify(host.clone(), device_repaired, host, host)
    assert result.passed


def test_meaningful_value_mismatch_fails():
    host = make_output()
    corrupted = host.clone()
    corrupted[0, 0, 0, 0] += 0.5
    result = verify(host.clone(), corrupted, host, host)
    assert not result.passed
    assert any("device" in reason for reason in result.failure_reasons)


def test_class_change_on_device_fails():
    """The exact shape of the backend bug this gate exists to catch."""
    host = make_output()
    transposed = host.transpose(2, 3).contiguous()
    result = verify(host.clone(), transposed, host, host)
    assert not result.passed
    assert result.argmax_agreement < 1.0


def test_a_backend_that_already_drifts_is_a_warning_not_a_repair_failure():
    """The Inception V3 case, and the reason this split exists.

    The backend does not reproduce its host result - and did not before the
    repair either. The repair itself is faithful: both device outputs agree
    exactly. Blaming the repair for the backend's pre-existing drift is what
    this used to do.
    """
    host = make_output()
    drift = 0.25
    result = verify(host + drift, host + drift, host, host)

    assert result.passed, "the repair did not change the device output"
    assert result.failure_reasons == []
    assert result.backend_fidelity == device_verification.BACKEND_FIDELITY_WARNING
    assert result.backend_fidelity_acceptable
    assert "before any repair" in result.backend_fidelity_reason


def test_device_repaired_disagreeing_with_host_fails():
    """Correct on the host, wrong on the device: the whole point of this check."""
    host_original = make_output(seed=0)
    host_repaired = host_original.clone()          # host says the repair is exact
    device_original = host_original.clone()
    device_repaired = host_original + 0.3          # the device disagrees

    result = verify(device_original, device_repaired, host_original, host_repaired)

    # Repair fidelity: the candidate changed what the device computes.
    assert not result.passed
    assert any("differs from the original" in r for r in result.failure_reasons)
    # And backend fidelity blames the right party: the original was clean.
    assert result.backend_fidelity == device_verification.BACKEND_FIDELITY_FAIL
    assert "the rewrite introduced this" in result.backend_fidelity_reason


def test_a_candidate_that_breaks_a_clean_backend_fails():
    """Original reproduced its host result; the candidate does not."""
    host_original = make_output(seed=0)
    # The candidate genuinely computes something different on host and device
    # alike, so repair fidelity has nothing to say - only backend fidelity does.
    host_repaired = host_original + 0.5
    result = verify(host_original.clone(), host_original + 0.5 + 0.4,
                    host_original, host_repaired)

    assert result.backend_fidelity == device_verification.BACKEND_FIDELITY_FAIL
    assert not result.backend_fidelity_acceptable


def test_a_candidate_dramatically_worse_than_a_drifting_baseline_fails():
    """A drifting baseline buys the right to be comparable, not arbitrary."""
    status, reason = device_verification.classify_backend_fidelity(
        original_error=1.3e-05, candidate_error=1.3e-03)
    assert status == device_verification.BACKEND_FIDELITY_FAIL
    assert "materially worse" in reason


def test_a_candidate_comparable_to_a_drifting_baseline_warns():
    """The measured DenseNet169 case: the candidate is slightly better."""
    status, _ = device_verification.classify_backend_fidelity(
        original_error=1.252e-05, candidate_error=1.222e-05)
    assert status == device_verification.BACKEND_FIDELITY_WARNING


def test_backend_fidelity_within_tolerance_is_silent():
    """The measured DenseNet121 case: nothing to report."""
    status, reason = device_verification.classify_backend_fidelity(
        original_error=6.44e-06, candidate_error=7.15e-06)
    assert status == device_verification.BACKEND_FIDELITY_OK
    assert reason == ""


def test_backend_fidelity_never_widens_the_tolerance():
    """The regression factor decides attribution, never what counts as equal."""
    from delegate_doctor import verification

    assert verification.MAX_ABSOLUTE_ERROR_TOLERANCE == 1e-5
    # Just past tolerance with no baseline to excuse it is still a failure,
    # however small the absolute number is.
    status, _ = device_verification.classify_backend_fidelity(
        original_error=0.0, candidate_error=1.01e-05)
    assert status == device_verification.BACKEND_FIDELITY_FAIL


def test_shape_mismatch_fails_before_comparing_values():
    host = make_output(shape=(1, 4, 8, 8))
    wrong_shape = make_output(shape=(1, 4, 4, 4))
    result = verify(host.clone(), wrong_shape, host, host)
    assert not result.passed
    assert any("shape" in reason for reason in result.failure_reasons)


def test_device_verification_uses_the_same_tolerance_as_host_verification():
    """One correctness policy, not two."""
    from delegate_doctor import verification

    host = make_output()
    just_inside = host + verification.MAX_ABSOLUTE_ERROR_TOLERANCE * 0.5
    just_outside = host + verification.MAX_ABSOLUTE_ERROR_TOLERANCE * 2.0

    assert verify(host.clone(), just_inside, host, host, dim=None).passed
    assert not verify(host.clone(), just_outside, host, host, dim=None).passed


# --- end to end, with the device mocked ------------------------------------

def test_run_device_verification_end_to_end(tmp_path, monkeypatch):
    host_original = make_output(seed=1)
    host_repaired = host_original.clone()

    fake = FakeAdb(tmp_path, tensor=host_original)
    fake.install(monkeypatch)

    result = run_device_verification(
        before_pte_path="before.pte",
        after_pte_path="after.pte",
        input_paths=["input.bin"],
        bench_runner_path="bench",
        original_host_output=host_original,
        repaired_host_output=host_repaired,
        output_dir=str(tmp_path),
        serial="emulator-5554",
        argmax_dim=1,
    )

    assert result.passed
    # the run leaves self-describing metadata beside the pulled bytes
    sidecar = tmp_path / "device_output.meta.txt"
    assert sidecar.is_file()
    assert "dtype=float32" in sidecar.read_text()


def test_result_serializes_for_the_run_artifacts(tmp_path, monkeypatch):
    host = make_output()
    fake = FakeAdb(tmp_path, tensor=host)
    fake.install(monkeypatch)

    result = run_device_verification(
        before_pte_path="b.pte", after_pte_path="a.pte", input_paths=["i.bin"],
        bench_runner_path="bench", original_host_output=host,
        repaired_host_output=host, output_dir=str(tmp_path),
        serial="emulator-5554", argmax_dim=1,
    )

    as_dict = result.to_dict()
    assert as_dict["passed"] is True
    assert as_dict["repaired_vs_original"]["max_absolute_error"] == 0.0
    assert as_dict["original_device_vs_host"] is not None
    assert as_dict["repaired_device_vs_host"] is not None
