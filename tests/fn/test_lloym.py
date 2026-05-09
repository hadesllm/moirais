"""Tests for moirais.fn.lloym — Lloyd-Max optimal quantizer."""

import numpy as np
import pytest

from moirais.fn.lloym import lloyd_max


class TestLloydMax:

    def test_returns_result(self):
        x = np.random.default_rng(42).standard_normal(200)
        res = lloyd_max(x, levels=8)
        assert res.name == "lloyd_max"
        assert res.value >= 0

    def test_centroid_count(self):
        x = np.random.default_rng(0).standard_normal(100)
        res = lloyd_max(x, levels=4)
        assert len(res.extra["centroids"]) == 4

    def test_more_levels_lower_mse(self):
        x = np.random.default_rng(1).standard_normal(200)
        mse4 = lloyd_max(x, levels=4).value
        mse16 = lloyd_max(x, levels=16).value
        assert mse16 <= mse4

    def test_labels_valid(self):
        x = np.random.default_rng(2).standard_normal(50)
        res = lloyd_max(x, levels=8)
        assert np.all(res.extra["labels"] >= 0)
        assert np.all(res.extra["labels"] < 8)
