"""Tests for moirais.fn.zsrbf -- Multiquadric RBF interpolation"""

import numpy as np
import pytest

from moirais.fn.zsrbf import rbf_multiquad


class TestRbfMultiquad:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rbf_multiquad(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = rbf_multiquad(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
