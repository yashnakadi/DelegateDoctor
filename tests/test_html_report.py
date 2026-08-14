"""The HTML report: self-contained, honest, escaped, and free of decoration.

Offline and fast. Most cases build an `OptimizationResult` directly rather than
running a model, because the report is a pure formatter - that is the whole
point of keeping it in its own module.
"""

import re
from pathlib import Path

import pytest

from delegate_doctor import html_report, result as result_module
from delegate_doctor.result import OptimizationResult


# --- small stand-ins for the measured objects -------------------------------

class FakeDelegation:
    def __init__(self, total=100, portable=1, blobs=1):
        self.total_ops = total
        self.portable_op_total = portable
        self.delegated_op_total = total - portable
        self.delegate_blob_count = blobs

    @property
    def operator_delegation_fraction(self):
        return self.delegated_op_total / self.total_ops


class FakeKernel:
    def __init__(self, name="_softmax.out", total_ms=48.2, fraction=0.634):
        self.name = f"native_call{name}"
        self.operator_name = name
        self.total_ms = total_ms
        self.call_count = 1
        self.runtime_fraction = fraction


class FakeProfile:
    def __init__(self, kernels=None, runtime_delegation=0.343):
        self.method_execute_ms = 76.0
        self.portable_ms = 50.0
        self.delegated_ms = 26.0
        self.delegate_call_count = 1
        self.operator_call_count = 3
        self.portable_kernels = kernels if kernels is not None else [FakeKernel()]
        self.accounting_warning = ""
        self._runtime = runtime_delegation

    @property
    def runtime_delegation_fraction(self):
        return self._runtime


class FakeMetrics:
    max_absolute_error = 1.863e-08
    mean_absolute_error = 3.2e-09
    mean_squared_error = 2.0e-17
    max_relative_error = 4.2e-07


class FakeVerification:
    def __init__(self, passed=True, argmax=1.0, reasons=None):
        self.passed = passed
        self.repaired_vs_original = FakeMetrics()
        self.argmax_agreement = argmax
        self.failure_reasons = reasons or []

    @property
    def status_text(self):
        return "PASS" if self.passed else "FAIL"


class FakeStats:
    def __init__(self, p50):
        self.p50_ms = p50
        self.p95_ms = p50 * 1.15
        self.mean_ms = p50 * 1.05
        self.sample_count = 450


class FakeBenchmark:
    def __init__(self, before=242.69, after=65.53):
        self.before = FakeStats(before)
        self.after = FakeStats(after)
        self.warmup_iterations = 20
        self.measured_iterations = 150
        self.repetitions = 3
        self.threads = 4
        self.device_description = "TestTarget (arm64-v8a, Android 10)"
        self.device_is_emulator = False

    @property
    def p50_speedup(self):
        return self.before.p50_ms / self.after.p50_ms


class FakeDecision:
    def __init__(self, accepted=True, headline="REPAIR ACCEPTED"):
        self.accepted = accepted
        self.headline = headline
        self.speedup = 3.79
        self.message = "..."


class FakeSite:
    def __init__(self, text="softmax(dim=1) on [1, 21, 256, 256]"):
        self.node_name = "softmax"
        self._text = text

    def explain(self):
        return self._text


class FakeDetection:
    def __init__(self, applies=True, sites=None, skipped=()):
        self.applies = applies
        self.detections = sites if sites is not None else [FakeSite()]
        self.skipped = list(skipped)


CATALOG = {
    "DD-001": {
        "title": "non-last-dimension softmax",
        "rewrite": "softmax(dim=D) -> view -> permute -> softmax(dim=-1)",
        "matches": lambda name: "softmax" in name,
        "flow_before": ("softmax(dim=1)",),
        "flow_after": ("view", "permute", "softmax(-1)", "permute", "view"),
    },
}


def build(status, **fields):
    outcome = OptimizationResult(status=status, repair_catalog=dict(CATALOG))
    outcome.model_name = fields.pop("model_name", "PSPNet")
    for name, value in fields.items():
        setattr(outcome, name, value)
    for name in result_module.STAGE_ORDER:
        if outcome.stage(name) is None:
            outcome.record(name, result_module.NOT_RUN)
    return outcome


def accepted_result():
    return build(
        result_module.REPAIR_ACCEPTED,
        before_delegation=FakeDelegation(total=125, portable=4),
        after_delegation=FakeDelegation(total=130, portable=0),
        before_profile=FakeProfile(),
        after_profile=FakeProfile(runtime_delegation=0.994),
        detections={"DD-001": FakeDetection()},
        repairs_applied={"DD-001": 1},
        host_verification=FakeVerification(),
        device_verification=FakeVerification(),
        benchmark=FakeBenchmark(),
        decision=FakeDecision(),
        device_description="RMX2030 · arm64-v8a · Android 10",
        run_dir="/tmp/run_042",
        output_pte="/tmp/run_042/optimized_model.pte",
    )


def render(outcome):
    return html_report.render(outcome, executorch_version="1.4.0")


def visible_text(page: str) -> str:
    """The page with tags stripped, for content assertions."""
    body = page[page.index("<body>"):]
    return re.sub(r"<[^>]+>", " ", body)


# --- structure and self-containment -----------------------------------------

def test_the_report_file_is_written(tmp_path):
    path = html_report.generate_html_report(accepted_result(), str(tmp_path))
    assert path.endswith("report.html")
    assert (tmp_path / "report.html").is_file()


def test_the_document_is_well_formed():
    page = render(accepted_result())
    assert page.startswith("<!DOCTYPE html>")
    assert '<html lang="en">' in page
    assert "<title>" in page and "</title>" in page
    assert page.rstrip().endswith("</html>")
    assert page.count("<body>") == page.count("</body>") == 1


@pytest.mark.parametrize("status", [
    result_module.REPAIR_ACCEPTED, result_module.REPAIR_REJECTED,
    result_module.FULLY_DELEGATED, result_module.NO_REPAIR_AVAILABLE,
    result_module.NO_REPAIR_REQUIRED, result_module.ANALYSIS_COMPLETE,
    result_module.DEVICE_EXECUTION_UNSUPPORTED,
    result_module.EXECUTORCH_LOWERING_UNSUPPORTED,
])
def test_every_status_renders(status):
    """No outcome may crash the formatter or produce an empty page."""
    page = render(build(status))
    assert page.startswith("<!DOCTYPE html>")
    assert len(page) > 1000


def test_the_report_has_no_remote_dependencies():
    page = render(accepted_result())
    assert not re.search(r"https?://", page), "the report references a remote URL"
    for token in ("<script", "<iframe", "cdn.", "googleapis", "unpkg",
                  "jsdelivr", "@import", "<img"):
        assert token not in page, f"the report depends on {token}"


def test_the_report_carries_its_own_styling():
    page = render(accepted_result())
    assert "<style>" in page
    assert 'rel="stylesheet"' not in page


def test_the_report_contains_no_emoji():
    """Typography and colour only - no pictographs anywhere."""
    page = render(accepted_result())
    for character in page:
        code = ord(character)
        assert not (0x1F000 <= code <= 0x1FAFF), f"emoji in report: {character!r}"
        assert not (0x2600 <= code <= 0x27BF), f"pictograph in report: {character!r}"
        assert code not in (0xFE0F, 0x2705, 0x274C), f"emoji in report: {character!r}"


def test_the_page_is_small_enough_to_open_instantly():
    assert len(render(accepted_result())) < 200_000


# --- repair accepted ---------------------------------------------------------

def test_an_accepted_repair_leads_with_the_speedup():
    page = render(accepted_result())
    text = visible_text(page)
    assert "3.70x" in text or "3.70" in text or "3.7" in text
    assert "FASTER" in text
    assert "REPAIR ACCEPTED" in text


def test_an_accepted_repair_shows_both_latencies():
    text = visible_text(render(accepted_result()))
    assert "242.69 ms" in text
    assert "65.53 ms" in text


def test_an_accepted_repair_shows_delegation_and_the_rule():
    text = visible_text(render(accepted_result()))
    assert "Operator-count delegation" in text
    assert "Runtime-weighted delegation" in text
    assert "34.3%" in text                      # runtime, before
    assert "DD-001" in text
    assert "non-last-dimension softmax" in text


def test_an_accepted_repair_shows_correctness():
    text = visible_text(render(accepted_result()))
    assert "Host verification" in text
    assert "Android verification" in text
    assert "1.863e-08" in text
    assert "100.00%" in text                    # argmax agreement


def test_the_target_is_named():
    text = visible_text(render(accepted_result()))
    assert "RMX2030" in text
    assert "Measured on a physical Android device" in text


def test_an_emulator_run_is_labelled_and_carries_the_caveat():
    """report.html is the shared artifact; the caveat must travel with it."""
    outcome = accepted_result()
    outcome.device_is_emulator = True
    outcome.device_description = "sdk_gphone64_arm64 · arm64-v8a · Android 15"
    text = visible_text(render(outcome))
    assert "Measured on an Android emulator" in text
    assert "not representative of physical Android hardware" in text


def test_a_shareable_report_never_carries_the_adb_serial():
    outcome = accepted_result()
    outcome.device_is_emulator = True
    outcome.device_description = "sdk_gphone64_arm64 · arm64-v8a · Android 15"
    page = render(outcome)
    for serial in ("emulator-5554", "a65d7d8b"):
        assert serial not in page




def test_the_repair_diagram_comes_from_rule_metadata():
    page = render(accepted_result())
    assert "softmax(-1)" in page                # from flow_after
    assert "permute" in page


def test_a_rule_without_diagram_metadata_renders_without_one():
    outcome = accepted_result()
    outcome.repair_catalog = {"DD-001": {"title": "t", "rewrite": "r",
                                         "matches": lambda n: False}}
    page = render(outcome)
    assert "DD-001" in page
    assert 'class="flow"' not in page


# --- healthy model -----------------------------------------------------------

def healthy_result():
    return build(
        result_module.FULLY_DELEGATED,
        model_name="MobileNetV2",
        before_delegation=FakeDelegation(total=100, portable=0),
        before_profile=FakeProfile(kernels=[], runtime_delegation=1.0),
        detections={},
    )


def test_a_healthy_model_reads_as_a_finding_not_an_empty_run():
    text = visible_text(render(healthy_result()))
    assert "DEPLOYMENT HEALTHY" in text
    assert "100.0%" in text
    assert "No portable hotspot detected" in text
    assert "found no meaningful fallback" in text


def test_a_healthy_model_invents_no_repair_or_benchmark():
    text = visible_text(render(healthy_result()))
    assert "Not required" in text
    assert "FASTER" not in text
    assert "Latency" not in text
    assert "Speedup" not in text


def test_a_healthy_model_says_delegation_is_fine():
    text = visible_text(render(healthy_result()))
    assert "No significant portable runtime bottleneck detected." in text


# --- no repair available -----------------------------------------------------

def test_no_matching_rule_still_shows_the_bottleneck():
    outcome = build(
        result_module.NO_REPAIR_AVAILABLE,
        model_name="SomeModel",
        before_delegation=FakeDelegation(total=100, portable=3),
        before_profile=FakeProfile(
            kernels=[FakeKernel("some_operator.out", 18.6, 0.213)],
            runtime_delegation=0.724),
        detections={},
    )
    text = visible_text(render(outcome))
    assert "NO REPAIR AVAILABLE" in text
    assert "some_operator.out" in text
    assert "18.60 ms" in text
    assert "21.3%" in text
    assert "No matching repair" in text
    assert "candidates for a future repair rule" in text
    # It must not imply anything was optimized.
    assert "FASTER" not in text


def test_extra_hotspots_are_summarized_not_dumped():
    kernels = [FakeKernel(f"op_{index}.out", 10 - index, 0.1) for index in range(15)]
    outcome = build(
        result_module.NO_REPAIR_AVAILABLE,
        before_delegation=FakeDelegation(),
        before_profile=FakeProfile(kernels=kernels),
        detections={},
    )
    page = render(outcome)
    text = visible_text(page)
    assert "+ 12 additional portable operator(s)" in text
    # The full list still exists, inside the collapsed section.
    assert "All portable operators" in page
    assert "op_14.out" in page


# --- rejected repair ---------------------------------------------------------

def test_a_rejected_repair_shows_the_regression_plainly():
    outcome = accepted_result()
    outcome.status = result_module.REPAIR_REJECTED
    outcome.benchmark = FakeBenchmark(before=100.0, after=119.0)
    outcome.decision = FakeDecision(
        accepted=False, headline="REPAIR REJECTED - no performance improvement")
    outcome.output_pte = None

    text = visible_text(render(outcome))
    assert "REPAIR REJECTED" in text
    assert "SLOWER" in text
    assert "19.0%" in text
    assert "100.00 ms" in text and "119.00 ms" in text
    assert "no performance improvement" in text


def test_a_failed_verification_is_shown_with_its_reason():
    outcome = accepted_result()
    outcome.status = result_module.REPAIR_REJECTED
    outcome.host_verification = FakeVerification(
        passed=False, reasons=["max abs error above tolerance"])
    text = visible_text(render(outcome))
    assert "FAIL" in text
    assert "max abs error above tolerance" in text


# --- device unavailable ------------------------------------------------------

def static_result():
    outcome = build(
        result_module.ANALYSIS_COMPLETE,
        before_delegation=FakeDelegation(total=41, portable=1),
        detections={"DD-001": FakeDetection()},
    )
    outcome.record(result_module.EXPORT, result_module.PASS)
    outcome.record(result_module.LOWERING, result_module.PASS)
    outcome.record(result_module.DELEGATION, result_module.PASS)
    outcome.record(result_module.DEVICE, result_module.UNSUPPORTED,
                   "input 0 is torch.int64; the Android input transport writes "
                   "raw fp32 blobs")
    return outcome


def test_a_static_run_shows_no_runtime_numbers():
    text = visible_text(render(static_result()))
    assert "NOT MEASURED" in text
    assert "Requires profiling on the Arm target" in text
    assert "FASTER" not in text
    assert "0.00 ms" not in text                # never a fake zero


def test_a_static_run_explains_why_the_device_stage_stopped():
    text = visible_text(render(static_result()))
    assert "torch.int64" in text
    assert "No measurement target" in text


def test_a_static_run_does_not_claim_a_repair_was_applied():
    text = visible_text(render(static_result()))
    assert "not applied" in text.lower()
    assert "Sites repaired" not in text


def test_stage_statuses_are_the_structured_ones():
    page = render(static_result())
    assert ">PASS<" in page
    assert ">UNSUPPORTED<" in page
    assert ">NOT RUN<" in page


def test_a_lowering_failure_renders_without_a_delegation_section():
    outcome = build(result_module.EXECUTORCH_LOWERING_UNSUPPORTED)
    outcome.record(result_module.EXPORT, result_module.PASS)
    outcome.record(result_module.LOWERING, result_module.FAILED,
                   "RuntimeError: edge dialect rejected aten.exotic")
    text = visible_text(render(outcome))
    assert "LOWERING UNSUPPORTED" in text
    assert "Delegation health" not in text
    assert "aten.exotic" in text


# --- escaping ----------------------------------------------------------------

HOSTILE = '<script>alert("x")</script> & < > " \''


def test_hostile_text_is_escaped_everywhere():
    outcome = accepted_result()
    outcome.model_name = HOSTILE
    outcome.device_description = HOSTILE
    outcome.run_dir = HOSTILE
    outcome.output_pte = HOSTILE
    outcome.detections = {"DD-001": FakeDetection(sites=[FakeSite(HOSTILE)])}
    outcome.repair_catalog = {
        "DD-001": {"title": HOSTILE, "rewrite": HOSTILE, "matches": lambda n: False}
    }
    outcome.host_verification = FakeVerification(passed=False, reasons=[HOSTILE])

    page = render(outcome)
    assert "<script>" not in page
    assert "alert(" not in page or "&quot;" in page
    assert "&lt;script&gt;" in page
    assert "&amp;" in page


def test_a_hostile_stage_detail_is_escaped():
    outcome = build(result_module.EXECUTORCH_LOWERING_UNSUPPORTED)
    outcome.record(result_module.LOWERING, result_module.FAILED, HOSTILE)
    page = render(outcome)
    assert "<script>" not in page
    assert "&lt;script&gt;" in page


def test_the_escape_helper_covers_the_dangerous_characters():
    escaped = html_report.esc('<a href="x">&</a>')
    for character in ("<", ">", '"'):
        assert character not in escaped
    assert "&lt;" in escaped and "&gt;" in escaped and "&amp;" in escaped


def test_escaping_handles_none_and_numbers():
    assert html_report.esc(None) == ""
    assert html_report.esc(3.5) == "3.5"


# --- values are formatted honestly -------------------------------------------

def test_bars_never_exceed_their_track():
    assert html_report._width(1.7) == "100.0"
    assert html_report._width(-0.2) == "0.0"
    assert html_report._width(None) == "0"


def test_missing_values_are_dashes_not_zeros():
    assert html_report._percent(None) == "—"
    assert html_report._ms(None) == "—"


def test_display_precision_matches_the_documented_convention():
    assert html_report._ms(242.6949) == "242.69 ms"
    assert html_report._percent(0.6187) == "61.9%"


# --- opening the report ------------------------------------------------------

def test_open_report_uses_the_standard_browser_boundary(tmp_path, monkeypatch):
    """The one place a browser is launched, and it is the stdlib's."""
    path = html_report.generate_html_report(accepted_result(), str(tmp_path))
    outcome = accepted_result()
    outcome.report_path = path

    opened = {}
    monkeypatch.setattr(result_module.webbrowser, "open",
                        lambda url: opened.setdefault("url", url) or True)

    assert outcome.open_report() is True
    assert opened["url"].startswith("file://")
    assert opened["url"].endswith("report.html")


def test_open_report_is_a_no_op_without_a_report(capsys, monkeypatch):
    monkeypatch.setattr(result_module.webbrowser, "open",
                        lambda url: pytest.fail("no browser should be launched"))
    outcome = accepted_result()
    outcome.report_path = None
    assert outcome.open_report() is False
    assert "No HTML report" in capsys.readouterr().out


def test_open_report_survives_a_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(result_module.webbrowser, "open",
                        lambda url: pytest.fail("no browser should be launched"))
    outcome = accepted_result()
    outcome.report_path = str(tmp_path / "gone.html")
    assert outcome.open_report() is False


def test_a_browser_failure_does_not_disturb_the_result(tmp_path, monkeypatch, capsys):
    """A headless machine must not turn a good analysis into an error."""
    path = html_report.generate_html_report(accepted_result(), str(tmp_path))
    outcome = accepted_result()
    outcome.report_path = path

    def explode(url):
        raise RuntimeError("no display")

    monkeypatch.setattr(result_module.webbrowser, "open", explode)

    assert outcome.open_report() is False
    assert path in capsys.readouterr().out
    assert outcome.status == result_module.REPAIR_ACCEPTED


def test_open_report_reports_a_refusing_browser(tmp_path, monkeypatch, capsys):
    path = html_report.generate_html_report(accepted_result(), str(tmp_path))
    outcome = accepted_result()
    outcome.report_path = path
    monkeypatch.setattr(result_module.webbrowser, "open", lambda url: False)

    assert outcome.open_report() is False
    assert "Could not open a browser" in capsys.readouterr().out


def test_the_report_path_is_a_plain_string_field():
    outcome = accepted_result()
    outcome.report_path = "/tmp/run_001/report.html"
    assert str(outcome.report_path).endswith("report.html")
    assert outcome.to_dict()["report_path"] == "/tmp/run_001/report.html"


# --- the CLI opens it by default ---------------------------------------------

class _ResolvedStub:
    """Stands in for a resolved model path when only dispatch is under test."""

    path = Path("model.py")
    kind = "python"
    from_workspace = False


class _FakeOutcome:
    """Stands in for a completed run when only report-opening is under test."""

    status = result_module.ANALYSIS_COMPLETE

    def __init__(self):
        self.opened = 0

    def open_report(self):
        self.opened += 1
        return True


@pytest.fixture
def cli_run(monkeypatch, tmp_path):
    """Run the CLI down to the report step without touching a model or device."""
    from delegate_doctor import cli

    outcome = _FakeOutcome()
    monkeypatch.setattr(cli.model_source, "resolve_model_input",
                        lambda target, root=".": _ResolvedStub())
    monkeypatch.setattr(cli, "ensure_target_available",
                        lambda *args, **kwargs: True)
    monkeypatch.setattr(cli, "prepare_model_source",
                        lambda path, **kwargs: object())
    monkeypatch.setattr(cli.pipeline, "run_optimization",
                        lambda spec, **options: outcome)
    return cli, outcome


def test_a_normal_cli_run_opens_the_report(cli_run):
    """The developer just watched a device benchmark; show them the result."""
    cli, outcome = cli_run
    assert cli.main(["optimize", "model.py"]) == 0
    assert outcome.opened == 1


def test_no_open_report_writes_it_without_opening_it(cli_run):
    cli, outcome = cli_run
    assert cli.main(["optimize", "model.py", "--no-open-report"]) == 0
    assert outcome.opened == 0


def test_a_non_interactive_run_never_opens_a_browser(cli_run):
    """A CI job launching a browser is a surprise, not a convenience."""
    cli, outcome = cli_run
    assert cli.main(["optimize", "model.py", "--non-interactive"]) == 0
    assert outcome.opened == 0


def test_the_python_api_never_opens_a_browser_on_its_own(monkeypatch):
    """optimize() runs inside other people's programs. It stays quiet."""
    from delegate_doctor import result as result_mod

    monkeypatch.setattr(result_mod.webbrowser, "open",
                        lambda url: pytest.fail("the API opened a browser"))
    outcome = accepted_result()
    outcome.report_path = None
    assert outcome.open_report() is False


def test_a_browser_failure_does_not_change_the_cli_exit_code(monkeypatch):
    """Opening the report is a courtesy; it cannot fail a completed analysis."""
    from delegate_doctor import cli

    class Refusing(_FakeOutcome):
        def open_report(self):
            raise RuntimeError("no display")

    outcome = Refusing()
    monkeypatch.setattr(cli.model_source, "resolve_model_input",
                        lambda target, root=".": _ResolvedStub())
    monkeypatch.setattr(cli, "ensure_target_available",
                        lambda *args, **kwargs: True)
    monkeypatch.setattr(cli, "prepare_model_source",
                        lambda path, **kwargs: object())
    monkeypatch.setattr(cli.pipeline, "run_optimization",
                        lambda spec, **options: outcome)

    assert cli.main(["optimize", "model.py"]) == 0
