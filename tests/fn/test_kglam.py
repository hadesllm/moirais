"""Tests for morie.fn.kglam -- Kriging weights (lambda)"""

import numpy as np
import pytest

from morie.fn.kglam import kriging_lambda


class TestKrigingLambda:
    def test_basic(self):
        vals = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = kriging_lambda(vals, x)
        assert result.statistic is not None

    def test_output_type(self):
        result = kriging_lambda(np.array([1.,2.,3.]), np.array([0.,1.,2.]))
        assert hasattr(result, "statistic")
