"""Tests for moirais.fn.cpads — load CPADS data."""

import pytest

from moirais.fn.cpads import cpads, load_cpads


def test_alias_is_same_function():
    """cpads and load_cpads are the same object."""
    assert cpads is load_cpads


def test_callable():
    """cpads is a callable."""
    assert callable(cpads)


def test_function_signature():
    """load_cpads accepts db_path and timeout kwargs."""
    import inspect
    sig = inspect.signature(load_cpads)
    params = list(sig.parameters.keys())
    assert "db_path" in params
    assert "timeout" in params
