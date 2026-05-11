"""Tests for morie.fn.vgxvg -- Cross-variogram estimation"""

import numpy as np
import pytest

from morie.fn.vgxvg import cross_vario


class TestCrossVario:
    def test_basic(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        values = rng.standard_normal(20)
        result = cross_vario(coords, values)
        assert result.statistic is not None

    def test_output_type(self):
        result = cross_vario(np.random.default_rng(0).uniform(0,1,(5,2)), np.ones(5))
        assert hasattr(result, "statistic")
