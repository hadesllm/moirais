"""Tests for morie.fn.kgstd -- Kriging standard error map"""

import numpy as np
import pytest

from morie.fn.kgstd import kriging_std_error


class TestKrigingStdError:
    def test_basic(self):
        vals = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = kriging_std_error(vals, x)
        assert result.statistic is not None

    def test_output_type(self):
        result = kriging_std_error(np.array([1.,2.,3.]), np.array([0.,1.,2.]))
        assert hasattr(result, "statistic")
