"""Tests for morie.fn.zsrgf -- Random Gaussian field"""

import numpy as np
import pytest

from morie.fn.zsrgf import random_gauss_field


class TestRandomGaussField:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = random_gauss_field(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = random_gauss_field(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
