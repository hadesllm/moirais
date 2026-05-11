"""Tests for agci (Agresti-Coull interval)."""
from morie.fn.agci import agresti_coull


def test_agresti_coull_basic():
    r = agresti_coull(successes=50, trials=100, alpha=0.05)
    assert 0.0 < r.extra["lower"] < r.extra["upper"] < 1.0


def test_agresti_coull_value():
    r = agresti_coull(successes=50, trials=100)
    assert abs(r.value - 0.5) < 0.1
