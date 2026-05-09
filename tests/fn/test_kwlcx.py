"""Tests for moirais.fn.kwlcx — kernel-smoothed Wilcoxon test."""

import numpy as np
import pytest

from moirais.fn.kwlcx import kwlcx


class TestKwlcx:
    def test_symmetric_not_rejected(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        res = kwlcx(data, mu0=0.0)
        assert res["p_value"] > 0.01

    def test_shifted_rejected(self):
        rng = np.random.default_rng(42)
        data = rng.normal(3, 1, 200)
        res = kwlcx(data, mu0=0.0)
        assert res["p_value"] < 0.05

    def test_output_structure(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        res = kwlcx(data)
        assert "statistic" in res
        assert "p_value" in res

    def test_raises_small(self):
        with pytest.raises(ValueError):
            kwlcx(np.array([1.0]))
