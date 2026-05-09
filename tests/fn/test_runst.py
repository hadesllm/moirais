"""Tests for runs_test."""
import numpy as np, pytest
from moirais.fn.runst import runs_test


class TestRunsTest:
    def test_random_data(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        r = runs_test(x)
        assert r.test_name == "Runs test"
        assert r.p_value > 0.05

    def test_patterned(self):
        x = np.array([1, 1, 1, 1, 10, 10, 10, 10] * 5, dtype=float)
        r = runs_test(x)
        assert r.p_value < 0.05

    def test_too_few(self):
        with pytest.raises(ValueError):
            runs_test([1.0])
