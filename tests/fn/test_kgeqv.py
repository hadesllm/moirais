"""Tests for moirais.fn.kgeqv -- Kriging equivalence to GLS"""

import numpy as np
import pytest

from moirais.fn.kgeqv import kriging_equiv


class TestKrigingEquiv:
    def test_basic(self):
        vals = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = kriging_equiv(vals, x)
        assert result.statistic is not None

    def test_output_type(self):
        result = kriging_equiv(np.array([1.,2.,3.]), np.array([0.,1.,2.]))
        assert hasattr(result, "statistic")
