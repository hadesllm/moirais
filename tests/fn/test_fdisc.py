"""Tests for moirais.fn.fdisc — f-divergence."""

import numpy as np
import pytest

from moirais.fn.fdisc import fdisc


class TestFdisc:
    def test_kl_identical(self):
        p = np.array([0.3, 0.7])
        result = fdisc(p, p, divergence="kl")
        assert result["divergence"] == pytest.approx(0.0, abs=1e-10)

    def test_kl_nonnegative(self):
        p = np.array([0.2, 0.8])
        q = np.array([0.5, 0.5])
        assert fdisc(p, q, divergence="kl")["divergence"] >= -1e-10

    def test_hellinger_symmetric(self):
        p = np.array([0.3, 0.7])
        q = np.array([0.6, 0.4])
        r1 = fdisc(p, q, divergence="hellinger")
        r2 = fdisc(q, p, divergence="hellinger")
        assert r1["divergence"] == pytest.approx(r2["divergence"], abs=1e-10)
        assert r1["symmetric"]

    def test_tv_bounded(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        result = fdisc(p, q, divergence="tv")
        assert result["divergence"] == pytest.approx(1.0, abs=1e-10)

    def test_js_bounded(self):
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        result = fdisc(p, q, divergence="js")
        assert 0 <= result["divergence"] <= np.log(2) + 1e-10

    def test_chi2(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        assert fdisc(p, q, divergence="chi2")["divergence"] == pytest.approx(0.0, abs=1e-10)

    def test_unknown_type(self):
        with pytest.raises(ValueError):
            fdisc(np.array([0.5, 0.5]), np.array([0.5, 0.5]), divergence="bogus")

    def test_invalid_pmf(self):
        with pytest.raises(ValueError):
            fdisc(np.array([0.3, 0.3]), np.array([0.5, 0.5]))
