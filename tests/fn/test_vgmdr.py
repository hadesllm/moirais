"""Tests for morie.fn.vgmdr -- Madogram estimator"""

import numpy as np
import pytest

from morie.fn.vgmdr import madogram


class TestMadogram:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = madogram(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = madogram(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
