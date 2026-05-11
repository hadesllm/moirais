"""Tests for morie.fn.xrgr2 -- Poisson gravity model"""

import numpy as np
import pytest

from morie.fn.xrgr2 import gravity_poisson


class TestGravityPoisson:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gravity_poisson(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = gravity_poisson(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
