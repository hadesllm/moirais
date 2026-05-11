"""Tests for morie.fn.renyh — Renyi entropy."""

import numpy as np
import pytest

from morie.fn.renyh import renyh


class TestRenyh:
    def test_alpha_1_is_shannon(self):
        pmf = np.array([0.3, 0.7])
        result = renyh(pmf, 1.0)
        assert result["entropy"] == pytest.approx(result["shannon_entropy"], abs=1e-10)

    def test_alpha_0_is_hartley(self):
        pmf = np.array([0.2, 0.3, 0.5])
        result = renyh(pmf, 0.0)
        assert result["entropy"] == pytest.approx(np.log2(3), abs=1e-10)

    def test_alpha_inf_is_min_entropy(self):
        pmf = np.array([0.5, 0.3, 0.2])
        result = renyh(pmf, np.inf)
        assert result["entropy"] == pytest.approx(-np.log2(0.5), abs=1e-10)

    def test_uniform_all_equal(self):
        pmf = np.array([0.25, 0.25, 0.25, 0.25])
        r0 = renyh(pmf, 0.5)["entropy"]
        r1 = renyh(pmf, 1.0)["entropy"]
        r2 = renyh(pmf, 2.0)["entropy"]
        assert r0 == pytest.approx(2.0, abs=1e-10)
        assert r1 == pytest.approx(2.0, abs=1e-10)
        assert r2 == pytest.approx(2.0, abs=1e-10)

    def test_decreasing_in_alpha(self):
        pmf = np.array([0.5, 0.3, 0.2])
        h_half = renyh(pmf, 0.5)["entropy"]
        h_2 = renyh(pmf, 2.0)["entropy"]
        assert h_half >= h_2 - 1e-10

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            renyh(np.array([0.5, 0.5]), -1.0)
