"""Tests for morie.fn.ptcrs -- Cross-type point pattern"""

import numpy as np
import pytest

from morie.fn.ptcrs import cross_pp


class TestCrossPp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = cross_pp(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = cross_pp(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
