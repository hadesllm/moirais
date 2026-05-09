"""Tests for biscn (bisection root finding)."""
import math
from moirais.fn.biscn import bisection_root


def test_bisection_sqrt2():
    r = bisection_root(lambda x: x**2 - 2, 1.0, 2.0)
    assert abs(r.value - math.sqrt(2)) < 1e-8


def test_bisection_sin():
    r = bisection_root(math.sin, 3.0, 4.0)
    assert abs(r.value - math.pi) < 1e-8
