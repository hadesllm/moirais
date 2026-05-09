"""Tests for moirais.fn.mvkov — Moving window variogram."""

import numpy as np
import pytest

from moirais.fn.mvkov import mvkov


class TestMvkov:

    def test_output_length(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (15, 2))
        values = rng.standard_normal(15)
        result = mvkov(coords, values, n_lags=5)
        assert len(result["local_variograms"]) == 15

    def test_lags_populated(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        values = rng.standard_normal(20)
        result = mvkov(coords, values)
        populated = [lv for lv in result["local_variograms"] if len(lv["lags"]) > 0]
        assert len(populated) > 0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="coords must be"):
            mvkov(np.ones((5, 3)), np.ones(5))
