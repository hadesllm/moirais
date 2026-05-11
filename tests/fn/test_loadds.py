"""Tests for morie.fn.loadds — load dataset from file."""

import pandas as pd
import pytest
import tempfile
import os

from morie.fn.loadds import loadds, load_dataset


def test_alias_is_same_function():
    """loadds and load_dataset are the same object."""
    assert loadds is load_dataset


def test_load_csv(tmp_path):
    """Loads a CSV file correctly."""
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df.to_csv(csv_path, index=False)
    result = loadds(csv_path)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert list(result.columns) == ["a", "b"]


def test_load_tsv(tmp_path):
    """Loads a TSV file correctly."""
    tsv_path = tmp_path / "test.tsv"
    df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
    df.to_csv(tsv_path, sep="\t", index=False)
    result = loadds(tsv_path)
    assert len(result) == 2


def test_load_json(tmp_path):
    """Loads a JSON file correctly."""
    json_path = tmp_path / "test.json"
    df = pd.DataFrame({"col": [1, 2, 3]})
    df.to_json(json_path)
    result = loadds(json_path)
    assert len(result) == 3


def test_file_not_found():
    """Raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        loadds("/nonexistent/path/data.csv")


def test_unsupported_extension(tmp_path):
    """Raises ValueError for unsupported file extension."""
    bad_path = tmp_path / "data.xyz"
    bad_path.write_text("stuff")
    with pytest.raises(ValueError, match="Unsupported"):
        loadds(bad_path)
