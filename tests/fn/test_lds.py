"""Tests for moirais.fn.lds — load dataset by catalog key."""

import pytest

from moirais.fn.lds import lds, load_dataset


def test_alias_is_same_function():
    """lds and load_dataset are the same object."""
    assert lds is load_dataset


def test_callable():
    """lds is a callable."""
    assert callable(lds)


def test_rejects_unknown_key():
    """Raises KeyError for an unknown dataset key."""
    with pytest.raises(KeyError, match="Unknown dataset key"):
        lds("totally_nonexistent_key_xyz")


def test_function_signature():
    """load_dataset accepts key, db_path, and timeout."""
    import inspect
    sig = inspect.signature(load_dataset)
    params = list(sig.parameters.keys())
    assert "key" in params
    assert "db_path" in params
    assert "timeout" in params
