"""Tests for morie.fn.kbwcv — LSCV bandwidth selection."""

import numpy as np
import pytest

from morie.fn.kbwcv import kbwcv


class TestKbwcv:
    def test_returns_positive_bw(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        res = kbwcv(data)
        assert res["bw_opt"] > 0

    def test_bw_reasonable_range(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        res = kbwcv(data)
        assert 0.05 < res["bw_opt"] < 2.0

    def test_grid_and_scores(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 50)
        res = kbwcv(data, n_bw=20)
        assert len(res["bw_grid"]) <= 20
        assert len(res["lscv_scores"]) == len(res["bw_grid"])

    def test_custom_range(self):
        data = np.arange(1, 51, dtype=float)
        res = kbwcv(data, bw_range=(0.5, 5.0))
        assert 0.5 <= res["bw_opt"] <= 5.0

    def test_raises_small(self):
        with pytest.raises(ValueError):
            kbwcv(np.array([1.0, 2.0]))
