"""Tests for morie.fn.ksgnr — kernel-smoothed sign test."""

import numpy as np
import pytest

from morie.fn.ksgnr import ksgnr


class TestKsgnr:
    def test_symmetric_data_not_rejected(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        res = ksgnr(data, mu0=0.0)
        assert res["p_value"] > 0.01

    def test_shifted_data_rejected(self):
        rng = np.random.default_rng(42)
        data = rng.normal(2, 1, 200)
        res = ksgnr(data, mu0=0.0)
        assert res["p_value"] < 0.05

    def test_output_keys(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        res = ksgnr(data)
        assert "statistic" in res
        assert "p_value" in res
        assert "mu0" in res
        assert "bw" in res

    def test_raises_small(self):
        with pytest.raises(ValueError):
            ksgnr(np.array([1.0]))
