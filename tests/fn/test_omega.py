"""Tests for morie.fn.omega — alias for omega-squared."""
from morie.fn.omega import omega


def test_omega_is_callable():
    result = omega(4.0, 2, 57, 60)
    assert isinstance(result, float)
    assert result >= 0


def test_omega_same_as_omega2():
    from morie.fn.omega2 import omega_squared
    assert omega(10.0, 3, 100, 104) == omega_squared(10.0, 3, 100, 104)
