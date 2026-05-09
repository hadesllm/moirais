"""Tests for breusch_pagan_test."""
import numpy as np, pytest
from moirais.fn.bpgtr import breusch_pagan_test


class TestBreuschPagan:
    def test_homoscedastic(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (60, 2))
        y = X @ np.array([1, 2]) + rng.normal(0, 1, 60)
        r = breusch_pagan_test(X, y)
        assert r.test_name == "Breusch-Pagan test"
        assert r.p_value > 0.05

    def test_too_few(self):
        with pytest.raises(ValueError):
            breusch_pagan_test(np.array([[1, 2], [3, 4]]), np.array([1, 2]))
