"""Tests for morie.fn.vgsil -- Sill estimation"""

import numpy as np
import pytest

from morie.fn.vgsil import sill_est


class TestSillEst:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sill_est(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = sill_est(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
