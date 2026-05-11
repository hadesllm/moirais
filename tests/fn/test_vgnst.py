"""Tests for morie.fn.vgnst -- Nested (composite) variogram"""

import numpy as np
import pytest

from morie.fn.vgnst import vario_nested


class TestVarioNested:
    def test_basic(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        values = rng.standard_normal(20)
        result = vario_nested(coords, values)
        assert result.statistic is not None

    def test_output_type(self):
        result = vario_nested(np.random.default_rng(0).uniform(0,1,(5,2)), np.ones(5))
        assert hasattr(result, "statistic")
