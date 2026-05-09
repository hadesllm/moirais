"""Tests for moirais.fn.zsrbg -- Gaussian RBF interpolation"""

import numpy as np
import pytest

from moirais.fn.zsrbg import rbf_gaussian


class TestRbfGaussian:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rbf_gaussian(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = rbf_gaussian(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
