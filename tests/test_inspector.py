"""Tests for moirais.inspector — output inspection and statistical verification."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from moirais.inspector import (
    InspectionResult,
    VerificationCheck,
    VerificationReport,
    inspect_output,
    inspect_directory,
    verify_statistical_output,
    verify_directory,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def good_csv(tmp_path):
    """CSV with valid statistical output."""
    df = pd.DataFrame({
        "term": ["intercept", "treatment", "age"],
        "estimate": [0.5, 1.2, -0.3],
        "se": [0.1, 0.2, 0.05],
        "p_value": [0.001, 0.04, 0.5],
        "ci_lower": [0.3, 0.8, -0.4],
        "ci_upper": [0.7, 1.6, -0.2],
        "odds_ratio": [1.6, 3.3, 0.7],
        "n": [500, 500, 500],
    })
    path = tmp_path / "good_output.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def bad_csv(tmp_path):
    """CSV with invalid statistical values."""
    df = pd.DataFrame({
        "term": ["a", "b"],
        "estimate": [1.0, 2.0],
        "se": [0.1, -0.5],           # negative SE
        "p_value": [0.05, 1.5],      # p > 1
        "ci_lower": [0.8, 2.5],      # lower > upper for row b
        "ci_upper": [1.2, 1.5],
        "odds_ratio": [2.7, -0.5],   # negative OR
        "n": [100, 100],
    })
    path = tmp_path / "bad_output.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def rsq_csv(tmp_path):
    """CSV with R-squared and AIC columns."""
    df = pd.DataFrame({
        "model": ["null", "full"],
        "r_squared": [0.0, 0.45],
        "aic": [1200.5, 980.3],
        "bic": [1210.0, 1000.1],
    })
    path = tmp_path / "model_fit.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def output_dir(tmp_path, good_csv):
    """Directory with multiple CSVs."""
    # good_csv is already in tmp_path
    df2 = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    (tmp_path / "simple.csv").write_text(df2.to_csv(index=False))
    return tmp_path


# ---------------------------------------------------------------------------
# InspectionResult
# ---------------------------------------------------------------------------


class TestInspectionResult:
    def test_inspect_good_csv(self, good_csv):
        result = inspect_output(good_csv)
        assert result.rows == 3
        assert result.columns == 8
        assert "estimate" in result.column_names
        assert result.missing_counts["estimate"] == 0
        assert result.summary_stats is not None
        assert len(result.head) == 3

    def test_inspect_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            inspect_output(tmp_path / "nonexistent.csv")

    def test_inspect_directory(self, output_dir):
        results = inspect_directory(output_dir)
        assert len(results) >= 2
        assert all(isinstance(r, InspectionResult) for r in results)

    def test_inspect_empty_dir(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        results = inspect_directory(tmp_path / "subdir")
        assert results == []

    def test_inspect_tsv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "data.tsv"
        df.to_csv(path, sep="\t", index=False)
        result = inspect_output(path)
        assert result.rows == 2
        assert result.columns == 2


# ---------------------------------------------------------------------------
# VerificationReport
# ---------------------------------------------------------------------------


class TestVerificationReport:
    def test_passed_when_all_ok(self):
        report = VerificationReport(
            file_path="test.csv",
            checks=[
                VerificationCheck("a", True, "ok"),
                VerificationCheck("b", True, "ok"),
            ],
        )
        assert report.passed is True
        assert report.error_count == 0

    def test_failed_on_error(self):
        report = VerificationReport(
            file_path="test.csv",
            checks=[
                VerificationCheck("a", True, "ok"),
                VerificationCheck("b", False, "bad", severity="error"),
            ],
        )
        assert report.passed is False
        assert report.error_count == 1

    def test_passed_with_warnings_only(self):
        report = VerificationReport(
            file_path="test.csv",
            checks=[
                VerificationCheck("a", True, "ok"),
                VerificationCheck("b", False, "warn", severity="warning"),
            ],
        )
        assert report.passed is True
        assert report.warning_count == 1


# ---------------------------------------------------------------------------
# verify_statistical_output — good data
# ---------------------------------------------------------------------------


class TestVerifyGoodData:
    def test_good_csv_passes(self, good_csv):
        report = verify_statistical_output(good_csv)
        assert report.passed is True
        assert report.error_count == 0

    def test_all_checks_present(self, good_csv):
        report = verify_statistical_output(good_csv)
        check_names = [c.name for c in report.checks]
        assert "file_readable" in check_names
        assert any("p_value" in n for n in check_names)
        assert any("se" in n for n in check_names)
        assert any("or" in n for n in check_names)
        assert "ci_order" in check_names


# ---------------------------------------------------------------------------
# verify_statistical_output — bad data
# ---------------------------------------------------------------------------


class TestVerifyBadData:
    def test_bad_csv_fails(self, bad_csv):
        report = verify_statistical_output(bad_csv)
        assert report.passed is False
        assert report.error_count > 0

    def test_catches_negative_se(self, bad_csv):
        report = verify_statistical_output(bad_csv)
        se_checks = [c for c in report.checks if "se_nonneg" in c.name]
        assert any(not c.passed for c in se_checks)

    def test_catches_p_value_out_of_range(self, bad_csv):
        report = verify_statistical_output(bad_csv)
        p_checks = [c for c in report.checks if "p_value_range" in c.name]
        assert any(not c.passed for c in p_checks)

    def test_catches_negative_or(self, bad_csv):
        report = verify_statistical_output(bad_csv)
        or_checks = [c for c in report.checks if "or_positive" in c.name]
        assert any(not c.passed for c in or_checks)

    def test_catches_ci_order(self, bad_csv):
        report = verify_statistical_output(bad_csv)
        ci_checks = [c for c in report.checks if c.name == "ci_order"]
        assert any(not c.passed for c in ci_checks)


# ---------------------------------------------------------------------------
# verify_statistical_output — edge cases
# ---------------------------------------------------------------------------


class TestVerifyEdgeCases:
    def test_missing_file(self, tmp_path):
        report = verify_statistical_output(tmp_path / "nope.csv")
        assert report.passed is False

    def test_empty_csv(self, tmp_path):
        path = tmp_path / "empty.csv"
        pd.DataFrame().to_csv(path, index=False)
        report = verify_statistical_output(path)
        # Empty file should not crash

    def test_r_squared_and_aic(self, rsq_csv):
        report = verify_statistical_output(rsq_csv)
        check_names = [c.name for c in report.checks]
        assert any("r_squared" in n for n in check_names)
        assert any("ic_finite" in n for n in check_names)
        assert report.passed is True

    def test_verify_directory(self, output_dir):
        reports = verify_directory(output_dir)
        assert len(reports) >= 2
        assert all(isinstance(r, VerificationReport) for r in reports)

    def test_nan_in_estimates(self, tmp_path):
        df = pd.DataFrame({
            "estimate": [1.0, float("nan")],
            "p_value": [0.05, 0.1],
        })
        path = tmp_path / "nan_est.csv"
        df.to_csv(path, index=False)
        report = verify_statistical_output(path)
        nan_checks = [c for c in report.checks if "no_nan" in c.name]
        assert any(not c.passed for c in nan_checks)
