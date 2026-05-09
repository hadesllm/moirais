"""Tests for moirais.fn.vrfydr — verify directory."""

import pandas as pd
import pytest

from moirais.fn.vrfydr import vrfydr, verify_directory
from moirais.inspector import VerificationReport


def test_alias_is_same_function():
    """vrfydr and verify_directory are the same object."""
    assert vrfydr is verify_directory


@pytest.fixture()
def output_dir(tmp_path):
    """Directory with two valid CSV files."""
    for name in ("a.csv", "b.csv"):
        df = pd.DataFrame({
            "estimate": [1.0, 2.0],
            "p_value": [0.05, 0.01],
            "se": [0.1, 0.2],
        })
        df.to_csv(tmp_path / name, index=False)
    return tmp_path


def test_returns_list_of_reports(output_dir):
    """verify_directory returns a list of VerificationReport."""
    results = vrfydr(output_dir)
    assert isinstance(results, list)
    assert all(isinstance(r, VerificationReport) for r in results)


def test_verifies_all_csvs(output_dir):
    """Verifies both CSV files in the directory."""
    results = vrfydr(output_dir)
    assert len(results) == 2


def test_not_a_directory():
    """Raises NotADirectoryError for a non-directory path."""
    with pytest.raises(NotADirectoryError):
        vrfydr("/nonexistent/directory/path")


def test_empty_directory(tmp_path):
    """Returns empty list for directory with no CSVs."""
    results = vrfydr(tmp_path)
    assert results == []
