"""Tests for ``moirais verify-pollution`` CLI (Workstream 6)."""

import json
import subprocess
import sys

import pytest


def _run_cli(*extra_args: str) -> subprocess.CompletedProcess:
    """Invoke the installed entrypoint as a subprocess to exercise argparse."""
    return subprocess.run(
        [sys.executable, "-m", "moirais.runner", "verify-pollution", *extra_args],
        capture_output=True, text=True,
    )


def test_verify_pollution_demo_no2_exit0():
    r = _run_cli(
        "--pollutant", "no2",
        "--demo",
        "--baseline-rate", "500",
        "--population", "2900000",
        "--reference", "10.0",
    )
    assert r.returncode == 0, f"stderr:\n{r.stderr}"
    assert "STATUS: ok" in r.stdout
    assert "Concentration-response" in r.stdout
    assert "RR:" in r.stdout
    assert "Attributable fraction (PAF):" in r.stdout


def test_verify_pollution_demo_pm25_exit0():
    r = _run_cli(
        "--pollutant", "pm25",
        "--demo",
        "--baseline-rate", "500",
        "--population", "1000000",
        "--reference", "5.0",
    )
    assert r.returncode == 0
    assert "PM25" in r.stdout


def test_verify_pollution_assumption_failure_exit1():
    # Exposure below reference → CRF monotonicity assumption fails
    r = _run_cli(
        "--pollutant", "no2",
        "--exposure-mean", "5",
        "--exposure-prevalence", "0.5",
        "--baseline-rate", "500",
        "--population", "1000000",
        "--reference", "10",
    )
    assert r.returncode == 1
    assert "assumption_failure" in r.stdout
    assert "FAIL" in r.stdout


def test_verify_pollution_prevalence_out_of_range_exit1():
    r = _run_cli(
        "--pollutant", "no2",
        "--exposure-mean", "25",
        "--exposure-prevalence", "1.5",
        "--baseline-rate", "500",
        "--population", "1000000",
    )
    assert r.returncode == 1
    assert "prevalence in [0,1]" in r.stdout


def test_verify_pollution_json_mode():
    r = _run_cli(
        "--pollutant", "no2",
        "--demo",
        "--baseline-rate", "500",
        "--population", "2900000",
        "--reference", "10.0",
        "--json",
    )
    assert r.returncode == 0
    payload = json.loads(r.stdout)
    assert payload["status"] == "ok"
    assert payload["pollutant"] == "no2"
    assert "pipeline" in payload
    assert "crf" in payload["pipeline"]
    assert payload["pipeline"]["crf"]["rr"] > 1.0
    assert 0.0 <= payload["pipeline"]["paf"] <= 1.0
    assert payload["pipeline"]["displaced"]["deaths_displaced"] > 0
    assert payload["pipeline"]["equity"] is not None


def test_verify_pollution_csv_missing_exposure_column_exit2(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("wrong_column\n1\n2\n3\n")
    r = _run_cli(
        "--pollutant", "no2",
        "--exposure-csv", str(p),
        "--baseline-rate", "500",
        "--population", "1000000",
        "--reference", "5.0",
    )
    assert r.returncode == 2
    assert "missing 'exposure' column" in r.stderr


def test_verify_pollution_unknown_pollutant_argparse_error():
    r = _run_cli(
        "--pollutant", "voc",
        "--demo",
    )
    # argparse-level validation → exit 2 (argparse uses 2 for usage errors)
    assert r.returncode == 2
    assert "invalid choice" in r.stderr


def test_verify_pollution_help_lists_command():
    r = subprocess.run(
        [sys.executable, "-m", "moirais.runner", "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "verify-pollution" in r.stdout
