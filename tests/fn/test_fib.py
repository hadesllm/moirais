"""Tests for fibonacci_ratio."""
import numpy as np, pytest
from moirais.fn.fib import fibonacci_ratio

class TestFib:
    def test_convergence(self):
        r = fibonacci_ratio(20)
        phi = (1 + np.sqrt(5)) / 2
        assert r.estimate == pytest.approx(phi, abs=1e-6)

    def test_error_decreases(self):
        r10 = fibonacci_ratio(10)
        r20 = fibonacci_ratio(20)
        assert r20.extra["abs_error"] < r10.extra["abs_error"]
