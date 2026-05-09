"""Tests for moirais.fn.dsinfo — dataset info."""

import pytest

from moirais.fn.dsinfo import dsinfo, dataset_info


def test_alias_is_same_function():
    """dsinfo and dataset_info are the same object."""
    assert dsinfo is dataset_info


def test_returns_dict_for_known_key():
    """Returns a dict with metadata for a known dataset key."""
    result = dsinfo("ocp21")
    assert isinstance(result, dict)
    assert result["key"] == "ocp21"
    assert "name" in result
    assert "source" in result


def test_rejects_unknown_key():
    """Raises KeyError for an unknown dataset key."""
    with pytest.raises(KeyError, match="Unknown dataset key"):
        dsinfo("nonexistent_xyz_999")


def test_includes_cache_status():
    """Result includes cached and cached_rows fields."""
    result = dsinfo("ocp21")
    assert "cached" in result
    assert "cached_rows" in result


def test_includes_local_exists():
    """Result includes local_exists field."""
    result = dsinfo("ocp21")
    assert "local_exists" in result
    assert isinstance(result["local_exists"], bool)
