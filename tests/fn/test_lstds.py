"""Tests for moirais.fn.lstds — list datasets."""

import pytest

from moirais.fn.lstds import lstds, list_datasets


def test_alias_is_same_function():
    """lstds and list_datasets are the same object."""
    assert lstds is list_datasets


def test_returns_list(tmp_path):
    """list_datasets returns a list of dicts."""
    db = tmp_path / "test.db"
    result = lstds(db_path=db)
    assert isinstance(result, list)
    assert len(result) > 0


def test_each_entry_has_expected_keys(tmp_path):
    """Each entry has key, name, source, survey, year, type, cached, rows."""
    db = tmp_path / "test.db"
    result = lstds(db_path=db)
    expected_keys = {"key", "name", "source", "survey", "year", "type", "cached", "rows"}
    for entry in result:
        assert expected_keys.issubset(entry.keys())


def test_known_datasets_present(tmp_path):
    """Known catalog entries appear in the listing."""
    db = tmp_path / "test.db"
    result = lstds(db_path=db)
    keys = {e["key"] for e in result}
    assert "ocp21" in keys
    assert "hibp" in keys
