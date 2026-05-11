"""Tests for morie.fn.vggau -- Gaussian variogram model"""

import numpy as np
import pytest

from morie.fn.vggau import vario_gaussian


class TestVarioGaussian:
    def test_basic(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        values = rng.standard_normal(20)
        result = vario_gaussian(coords, values)
        assert result.statistic is not None

    def test_output_type(self):
        result = vario_gaussian(np.random.default_rng(0).uniform(0,1,(5,2)), np.ones(5))
        assert hasattr(result, "statistic")
