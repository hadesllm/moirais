"""Tests for moirais.fn.rvrfy — render verification."""

import pytest

from moirais.fn.rvrfy import rvrfy, render_verification
from moirais.inspector import VerificationReport, VerificationCheck


def test_alias_is_same_function():
    """rvrfy and render_verification are the same object."""
    assert rvrfy is render_verification


def test_callable():
    """render_verification is callable."""
    assert callable(rvrfy)


def test_renders_passing_report():
    """render_verification runs without error on a passing report."""
    report = VerificationReport(
        file_path="test.csv",
        checks=[
            VerificationCheck("file_readable", True, "3 rows, 2 columns"),
            VerificationCheck("p_value_range:p", True, "3 values OK"),
        ],
    )
    # Should not raise; output goes to stdout (plain text in non-TTY)
    rvrfy(report)


def test_renders_failing_report():
    """render_verification runs without error on a failing report."""
    report = VerificationReport(
        file_path="bad.csv",
        checks=[
            VerificationCheck("file_readable", True, "1 row, 1 column"),
            VerificationCheck("p_value_range:p", False, "1/1 values outside [0,1]"),
        ],
    )
    rvrfy(report)
