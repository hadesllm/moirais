"""Tests for morie.fn.vgmat -- Matern variogram model"""

import numpy as np
import pytest

from morie.fn.vgmat import vario_matern


class TestVarioMatern:
    def test_basic(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        values = rng.standard_normal(20)
        result = vario_matern(coords, values)
        assert result.statistic is not None

    def test_output_type(self):
        result = vario_matern(np.random.default_rng(0).uniform(0,1,(5,2)), np.ones(5))
        assert hasattr(result, "statistic")
