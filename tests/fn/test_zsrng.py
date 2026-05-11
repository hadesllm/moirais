"""Tests for morie.fn.zsrng -- Random non-Gaussian field"""

import numpy as np
import pytest

from morie.fn.zsrng import random_nongauss


class TestRandomNongauss:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = random_nongauss(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = random_nongauss(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
