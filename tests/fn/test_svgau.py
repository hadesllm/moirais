"""Tests for morie.fn.svgau -- Gaussian spatial utility function"""

import numpy as np
import pytest

from morie.fn.svgau import gauss_utility


class TestGaussUtility:
    def test_basic(self):
        x = np.array([1.0, 2.0])
        result = gauss_utility(x, ideal_point=np.array([0.0, 0.0]))
        assert result.value is not None
        assert result.value >= 0

    def test_output_type(self):
        result = gauss_utility(np.array([1.0, 2.0]))
        assert hasattr(result, "value")
