"""Tests for morie.fn.clist — cache list."""

import pandas as pd
import pytest

from morie.fn.clist import clist, cache_list
from morie.fn.cstor import cstor


def test_alias_is_same_function():
    """clist and cache_list are the same object."""
    assert clist is cache_list


def test_empty_db_has_metadata_only(tmp_path):
    """A fresh cache DB has only the _morie_metadata table."""
    db = tmp_path / "empty.db"
    result = clist(db)
    tables = [r["table"] for r in result]
    assert "_morie_metadata" in tables


def test_lists_stored_tables(tmp_path):
    """Lists tables that have been stored."""
    db = tmp_path / "test.db"
    cstor(pd.DataFrame({"a": [1, 2]}), "alpha", db)
    cstor(pd.DataFrame({"b": [3, 4, 5]}), "beta", db)
    result = clist(db)
    table_map = {r["table"]: r["rows"] for r in result}
    assert "alpha" in table_map
    assert table_map["alpha"] == 2
    assert "beta" in table_map
    assert table_map["beta"] == 3


def test_returns_list_of_dicts(tmp_path):
    """Returns a list of dicts with 'table' and 'rows' keys."""
    db = tmp_path / "test.db"
    result = clist(db)
    assert isinstance(result, list)
    for item in result:
        assert "table" in item
        assert "rows" in item
