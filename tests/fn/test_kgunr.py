"""Tests for moirais.fn.kgunr -- Universal kriging residual"""

import numpy as np
import pytest

from moirais.fn.kgunr import uk_residual


class TestUkResidual:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = uk_residual(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = uk_residual(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
