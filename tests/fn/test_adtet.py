"""Tests for anderson_darling_test."""
import numpy as np, pytest
from moirais.fn.adtet import anderson_darling_test


class TestAndersonDarling:
    def test_normal_data(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        r = anderson_darling_test(x)
        assert r.test_name == "Anderson-Darling test"
        assert not r.extra["reject_at_5pct"]

    def test_uniform_rejects(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 200)
        r = anderson_darling_test(x)
        assert r.extra["reject_at_5pct"]

    def test_too_few(self):
        with pytest.raises(ValueError):
            anderson_darling_test([1, 2])
