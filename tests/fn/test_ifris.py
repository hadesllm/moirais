"""Tests for moirais.fn.ifris -- infection fatality rate."""

import numpy as np
import pytest
from moirais.fn.ifris import infection_fatality_rate


class TestIFR:
    def test_perfect_test(self):
        res = infection_fatality_rate(100, 100000, 0.10)
        assert res["ifr"] == pytest.approx(100 / 10000, rel=0.01)
        assert res["adjusted_seroprevalence"] == pytest.approx(0.10)

    def test_imperfect_test(self):
        res = infection_fatality_rate(100, 100000, 0.12,
                                     sensitivity=0.95, specificity=0.99)
        assert 0 < res["adjusted_seroprevalence"] <= 0.15

    def test_ci_ordering(self):
        res = infection_fatality_rate(50, 50000, 0.05)
        assert res["ci_lower"] <= res["ifr"] <= res["ci_upper"]

    def test_zero_sero(self):
        res = infection_fatality_rate(10, 100000, 0.0)
        assert res["ifr"] == np.inf

    def test_invalid_sero_raises(self):
        with pytest.raises(ValueError):
            infection_fatality_rate(10, 1000, 1.5)

    def test_bad_sensitivity_raises(self):
        with pytest.raises(ValueError):
            infection_fatality_rate(10, 1000, 0.1, sensitivity=0.0)
