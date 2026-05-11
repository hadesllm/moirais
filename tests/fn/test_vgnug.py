"""Tests for morie.fn.vgnug -- Nugget effect estimation"""

import numpy as np
import pytest

from morie.fn.vgnug import nugget_est


class TestNuggetEst:
    def test_basic(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        values = rng.standard_normal(20)
        result = nugget_est(coords, values)
        assert result.statistic is not None

    def test_output_type(self):
        result = nugget_est(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
