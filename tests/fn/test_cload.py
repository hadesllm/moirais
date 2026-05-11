"""Tests for morie.fn.cload — cache load."""

import pandas as pd
import pytest

from morie.fn.cload import cload, cache_load
from morie.fn.cstor import cstor


def test_alias_is_same_function():
    """cload and cache_load are the same object."""
    assert cload is cache_load


def test_returns_none_for_missing_table(tmp_path):
    """Returns None when the table does not exist."""
    db = tmp_path / "empty.db"
    result = cload("nonexistent", db)
    assert result is None


def test_loads_stored_data(tmp_path):
    """Loads data that was previously stored."""
    db = tmp_path / "test.db"
    df = pd.DataFrame({"col": [1, 2, 3]})
    cstor(df, "my_table", db)
    loaded = cload("my_table", db)
    assert isinstance(loaded, pd.DataFrame)
    assert len(loaded) == 3


def test_preserves_values(tmp_path):
    """Loaded values match stored values."""
    db = tmp_path / "test.db"
    df = pd.DataFrame({"val": [10, 20, 30]})
    cstor(df, "vals", db)
    loaded = cload("vals", db)
    assert list(loaded["val"]) == [10, 20, 30]
