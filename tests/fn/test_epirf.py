"""Tests for morie.fn.epirf -- effective Rt (Wallinga-Teunis)."""

import numpy as np
import pytest
from morie.fn.epirf import effective_rt


class TestEffectiveRt:
    def test_constant_incidence(self):
        inc = np.ones(30) * 10
        si = np.array([0.0, 0.3, 0.5, 0.2])
        res = effective_rt(inc, si, tau=1)
        mid = res["Rt"][10:20]
        assert np.nanmean(mid) == pytest.approx(1.0, abs=0.3)

    def test_output_shape(self):
        inc = np.array([1, 2, 5, 10, 15, 20, 18, 12, 6, 3])
        si = np.array([0.2, 0.5, 0.3])
        res = effective_rt(inc, si)
        assert len(res["Rt"]) == len(inc)
        assert len(res["ci_lower"]) == len(inc)

    def test_short_array_raises(self):
        with pytest.raises(ValueError):
            effective_rt(np.array([5]), np.array([0.5, 0.5]))

    def test_ci_nonnegative(self):
        rng = np.random.default_rng(42)
        inc = rng.poisson(10, 50).astype(float)
        si = np.array([0.1, 0.3, 0.4, 0.2])
        res = effective_rt(inc, si)
        assert np.all(res["ci_lower"][~np.isnan(res["ci_lower"])] >= 0)
