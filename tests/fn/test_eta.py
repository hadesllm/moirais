"""Tests for morie.fn.eta — alias for eta-squared."""
from morie.fn.eta import eta


def test_eta_is_callable():
    result = eta(4.0, 2, 57)
    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_eta_same_as_eta2():
    from morie.fn.eta2 import eta_squared
    assert eta(10.0, 3, 100) == eta_squared(10.0, 3, 100)
