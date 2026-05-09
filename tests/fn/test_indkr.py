"""Tests for moirais.fn.indkr — Indicator kriging."""

import numpy as np
import pytest

from moirais.fn.indkr import indkr


class TestIndkr:

    def test_probabilities_in_range(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        values = rng.standard_normal(20)
        target = rng.uniform(2, 8, (5, 2))
        result = indkr(coords, values, target)
        assert np.all(result["probabilities"] >= 0)
        assert np.all(result["probabilities"] <= 1)

    def test_custom_threshold(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (15, 2))
        values = rng.standard_normal(15)
        result = indkr(coords, values, coords[:3], threshold=0.0)
        assert result["threshold"] == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="coords must be"):
            indkr(np.ones((5, 2)), np.ones(3), np.ones((1, 2)))
