import json
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_eval_report.py"
SPEC = importlib.util.spec_from_file_location("check_eval_report", SCRIPT_PATH)
assert SPEC and SPEC.loader
check_eval_report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_eval_report)

check_report = check_eval_report.check_report
main = check_eval_report.main


def _write_report(path, metrics):
    path.write_text(
        json.dumps(
            {
                "dataset_name": "smoke",
                "case_count": 10,
                "metrics": metrics,
            }
        ),
        encoding="utf-8",
    )


def test_check_eval_report_passes_when_thresholds_are_met(tmp_path, capsys):
    report = tmp_path / "eval-report.json"
    _write_report(
        report,
        {
            "exact_top1_rate": 0.9,
            "effective_top1_rate": 1.0,
            "serious_top1_error_rate": 0.0,
        },
    )

    code = main([str(report), "--min-exact-top1", "0.88", "--min-effective-top1", "1.0"])

    assert code == 0
    output = capsys.readouterr().out
    assert "PASS exact_top1" in output
    assert "PASS effective_top1" in output
    assert "PASS serious_mismatch" in output


def test_check_eval_report_uses_exact_hit_at_1_fallback(tmp_path):
    report = tmp_path / "eval-report.json"
    _write_report(
        report,
        {
            "exact_hit_at_1": 0.91,
            "acceptable_hit_at_1": 1.0,
            "serious_top1_error_rate": 0.0,
        },
    )

    code = main([str(report), "--min-exact-top1", "0.9"])

    assert code == 0


def test_check_eval_report_fails_when_exact_top1_is_too_low(tmp_path, capsys):
    report = tmp_path / "eval-report.json"
    _write_report(
        report,
        {
            "exact_top1_rate": 0.7,
            "effective_top1_rate": 1.0,
            "serious_top1_error_rate": 0.0,
        },
    )

    code = main([str(report), "--min-exact-top1", "0.88"])

    assert code == 1
    assert "FAIL exact_top1" in capsys.readouterr().out


def test_check_eval_report_fails_when_effective_top1_is_too_low(tmp_path, capsys):
    report = tmp_path / "eval-report.json"
    _write_report(
        report,
        {
            "exact_top1_rate": 0.9,
            "effective_top1_rate": 0.97,
            "serious_top1_error_rate": 0.0,
        },
    )

    code = main([str(report), "--min-effective-top1", "1.0"])

    assert code == 1
    assert "FAIL effective_top1" in capsys.readouterr().out


def test_check_eval_report_fails_when_serious_mismatch_is_too_high(tmp_path, capsys):
    report = tmp_path / "eval-report.json"
    _write_report(
        report,
        {
            "exact_top1_rate": 0.9,
            "effective_top1_rate": 1.0,
            "serious_top1_error_rate": 0.02,
        },
    )

    code = main([str(report), "--max-serious-mismatch", "0.0"])

    assert code == 1
    assert "FAIL serious_mismatch" in capsys.readouterr().out


def test_check_eval_report_fails_when_required_metric_is_missing(tmp_path, capsys):
    report = tmp_path / "eval-report.json"
    _write_report(
        report,
        {
            "exact_top1_rate": 0.9,
            "effective_top1_rate": 1.0,
        },
    )

    code = main([str(report)])

    assert code == 2
    assert "missing numeric metric: serious_top1_error_rate" in capsys.readouterr().err


def test_check_report_returns_structured_json_summary():
    summary = check_report(
        {
            "dataset_name": "smoke",
            "case_count": 1,
            "metrics": {
                "exact_top1_rate": 0.9,
                "effective_top1_rate": 1.0,
                "serious_top1_error_rate": 0.0,
            },
        },
        min_exact_top1=0.88,
        min_effective_top1=1.0,
        max_serious_mismatch=0.0,
    )

    assert summary["ok"] is True
    assert [check["name"] for check in summary["checks"]] == [
        "exact_top1",
        "effective_top1",
        "serious_mismatch",
    ]
