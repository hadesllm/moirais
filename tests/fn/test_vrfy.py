"""Tests for morie.fn.vrfy — verify statistical output."""

import pandas as pd
import pytest

from morie.fn.vrfy import vrfy, verify_statistical_output
from morie.inspector import VerificationReport


def test_alias_is_same_function():
    """vrfy and verify_statistical_output are the same object."""
    assert vrfy is verify_statistical_output


@pytest.fixture()
def valid_csv(tmp_path):
    """CSV with valid statistical output."""
    df = pd.DataFrame({
        "estimate": [1.5, 2.3, -0.4],
        "se": [0.3, 0.5, 0.2],
        "p_value": [0.01, 0.05, 0.80],
        "ci_lower": [0.9, 1.3, -0.8],
        "ci_upper": [2.1, 3.3, 0.0],
    })
    path = tmp_path / "valid.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def bad_pval_csv(tmp_path):
    """CSV with an out-of-range p-value."""
    df = pd.DataFrame({
        "estimate": [1.0],
        "p_value": [1.5],  # invalid: > 1
    })
    path = tmp_path / "bad_pval.csv"
    df.to_csv(path, index=False)
    return path


def test_returns_verification_report(valid_csv):
    """verify_statistical_output returns a VerificationReport."""
    result = vrfy(valid_csv)
    assert isinstance(result, VerificationReport)


def test_valid_file_passes(valid_csv):
    """Valid statistical output passes all checks."""
    result = vrfy(valid_csv)
    assert result.passed


def test_bad_pvalue_fails(bad_pval_csv):
    """Out-of-range p-value causes a check failure."""
    result = vrfy(bad_pval_csv)
    failed = [c for c in result.checks if not c.passed and "p_value" in c.name]
    assert len(failed) > 0


def test_file_not_found():
    """Missing file produces a failed file_exists check."""
    result = vrfy("/nonexistent/path/data.csv")
    assert not result.passed


def test_negative_se_fails(tmp_path):
    """Negative standard error causes a check failure."""
    df = pd.DataFrame({"se": [-0.5], "estimate": [1.0]})
    path = tmp_path / "neg_se.csv"
    df.to_csv(path, index=False)
    result = vrfy(path)
    failed = [c for c in result.checks if not c.passed and "se_nonneg" in c.name]
    assert len(failed) > 0


def test_ci_order_check(tmp_path):
    """CI with lower > upper causes a check failure."""
    df = pd.DataFrame({
        "ci_lower": [5.0],
        "ci_upper": [1.0],
    })
    path = tmp_path / "bad_ci.csv"
    df.to_csv(path, index=False)
    result = vrfy(path)
    failed = [c for c in result.checks if not c.passed and "ci_order" in c.name]
    assert len(failed) > 0
