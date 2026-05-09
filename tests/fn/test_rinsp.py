"""Tests for moirais.fn.rinsp — render inspection."""

import pandas as pd
import pytest

from moirais.fn.rinsp import rinsp, render_inspection
from moirais.inspector import InspectionResult


def test_alias_is_same_function():
    """rinsp and render_inspection are the same object."""
    assert rinsp is render_inspection


def test_callable():
    """render_inspection is callable."""
    assert callable(rinsp)


def test_renders_without_error():
    """render_inspection runs without error on a valid InspectionResult."""
    result = InspectionResult(
        file_path="test.csv",
        rows=3,
        columns=2,
        column_names=["a", "b"],
        dtypes={"a": "int64", "b": "float64"},
        missing_counts={"a": 0, "b": 1},
        head=pd.DataFrame({"a": [1, 2, 3], "b": [4.0, None, 6.0]}),
        summary_stats=None,
    )
    # Should not raise; output goes to stdout (plain text in non-TTY)
    rinsp(result)
