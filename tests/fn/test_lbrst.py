"""Tests for lilliefors_test."""
import numpy as np, pytest
from moirais.fn.lbrst import lilliefors_test


class TestLilliefors:
    def test_normal_data(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        r = lilliefors_test(x)
        assert r.test_name == "Lilliefors test"
        assert not r.extra["reject_at_5pct"]

    def test_too_few(self):
        with pytest.raises(ValueError):
            lilliefors_test([1, 2, 3])
