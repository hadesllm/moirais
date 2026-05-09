"""Tests for moirais.fn.inspdr — inspect directory."""

import pandas as pd
import pytest

from moirais.fn.inspdr import inspdr, inspect_directory
from moirais.inspector import InspectionResult


def test_alias_is_same_function():
    """inspdr and inspect_directory are the same object."""
    assert inspdr is inspect_directory


@pytest.fixture()
def output_dir(tmp_path):
    """Directory with two CSV files."""
    for name in ("results_a.csv", "results_b.csv"):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        df.to_csv(tmp_path / name, index=False)
    return tmp_path


def test_returns_list_of_results(output_dir):
    """inspect_directory returns a list of InspectionResult."""
    results = inspdr(output_dir)
    assert isinstance(results, list)
    assert all(isinstance(r, InspectionResult) for r in results)


def test_finds_all_csvs(output_dir):
    """Finds both CSV files in the directory."""
    results = inspdr(output_dir)
    assert len(results) == 2


def test_not_a_directory():
    """Raises NotADirectoryError for a non-directory path."""
    with pytest.raises(NotADirectoryError):
        inspdr("/nonexistent/directory/path")


def test_empty_directory(tmp_path):
    """Returns empty list for directory with no CSVs."""
    results = inspdr(tmp_path)
    assert results == []
