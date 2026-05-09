"""Tests for moirais.fn.inspct — inspect output file."""

import pandas as pd
import pytest

from moirais.fn.inspct import inspct, inspect_output
from moirais.inspector import InspectionResult


def test_alias_is_same_function():
    """inspct and inspect_output are the same object."""
    assert inspct is inspect_output


@pytest.fixture()
def stat_csv(tmp_path):
    """Create a temporary CSV with statistical output."""
    df = pd.DataFrame({
        "estimate": [1.5, 2.3, -0.4],
        "se": [0.3, 0.5, 0.2],
        "p_value": [0.01, 0.05, 0.80],
        "ci_lower": [0.9, 1.3, -0.8],
        "ci_upper": [2.1, 3.3, 0.0],
    })
    path = tmp_path / "results.csv"
    df.to_csv(path, index=False)
    return path


def test_returns_inspection_result(stat_csv):
    """inspect_output returns an InspectionResult."""
    result = inspct(stat_csv)
    assert isinstance(result, InspectionResult)


def test_correct_shape(stat_csv):
    """Row and column counts are correct."""
    result = inspct(stat_csv)
    assert result.rows == 3
    assert result.columns == 5


def test_column_names(stat_csv):
    """Column names are captured."""
    result = inspct(stat_csv)
    assert "estimate" in result.column_names
    assert "p_value" in result.column_names


def test_dtypes_populated(stat_csv):
    """Dtype dict is populated for all columns."""
    result = inspct(stat_csv)
    assert len(result.dtypes) == 5


def test_missing_counts(stat_csv):
    """Missing counts are zero for complete data."""
    result = inspct(stat_csv)
    for col, count in result.missing_counts.items():
        assert count == 0


def test_file_not_found():
    """Raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        inspct("/nonexistent/path/data.csv")


def test_head_has_rows(stat_csv):
    """Head DataFrame is populated."""
    result = inspct(stat_csv)
    assert len(result.head) > 0
