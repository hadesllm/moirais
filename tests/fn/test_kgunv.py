"""Tests for moirais.fn.kgunv -- Universal kriging variance"""

import numpy as np
import pytest

from moirais.fn.kgunv import uk_variance


class TestUkVariance:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = uk_variance(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = uk_variance(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
