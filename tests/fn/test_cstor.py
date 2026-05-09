"""Tests for moirais.fn.cstor — cache store."""

import pandas as pd
import pytest

from moirais.fn.cstor import cstor, cache_store
from moirais.fn.cload import cload


def test_alias_is_same_function():
    """cstor and cache_store are the same object."""
    assert cstor is cache_store


def test_stores_and_returns_count(tmp_path):
    """cache_store stores a DataFrame and returns row count."""
    db = tmp_path / "test.db"
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    n = cstor(df, "test_table", db)
    assert n == 3


def test_stored_data_is_loadable(tmp_path):
    """Data stored via cstor can be loaded back."""
    db = tmp_path / "test.db"
    df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
    cstor(df, "roundtrip", db)
    loaded = cload("roundtrip", db)
    assert loaded is not None
    assert len(loaded) == 2
    assert list(loaded.columns) == ["x", "y"]


def test_replaces_existing_table(tmp_path):
    """Storing to the same table replaces the old data."""
    db = tmp_path / "test.db"
    df1 = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"a": [2, 3]})
    cstor(df1, "tbl", db)
    cstor(df2, "tbl", db)
    loaded = cload("tbl", db)
    assert len(loaded) == 2
