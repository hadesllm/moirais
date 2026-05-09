"""Tests for moirais.fn.quadu — quadratic utility."""
import numpy as np
from moirais.fn.quadu import quadu


def test_quadu_smoke():
    r = quadu(0.0, 1.0)
    assert r.name == "quadratic_utility"
    assert r.value < 0
    assert "beta" in r.extra


def test_quadu_zero_distance():
    r = quadu(2.0, 2.0)
    assert r.value == 0.0


def test_quadu_multidim():
    r = quadu([0, 0], [1, 1])
    assert r.value == -2.0
