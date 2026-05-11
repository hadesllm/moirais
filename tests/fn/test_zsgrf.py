"""Tests for morie.fn.zsgrf -- Focal grid statistics"""

import numpy as np
import pytest

from morie.fn.zsgrf import grid_focal


class TestGridFocal:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = grid_focal(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = grid_focal(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
