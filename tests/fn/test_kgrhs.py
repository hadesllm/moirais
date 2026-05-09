"""Tests for moirais.fn.kgrhs -- Kriging right-hand side vector"""

import numpy as np
import pytest

from moirais.fn.kgrhs import kriging_rhs


class TestKrigingRhs:
    def test_basic(self):
        vals = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = kriging_rhs(vals, x)
        assert result.statistic is not None

    def test_output_type(self):
        result = kriging_rhs(np.array([1.,2.,3.]), np.array([0.,1.,2.]))
        assert hasattr(result, "statistic")
