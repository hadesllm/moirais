"""Tests for morie.fn.ptmrk -- Marked point pattern analysis"""

import numpy as np
import pytest

from morie.fn.ptmrk import marked_pp


class TestMarkedPp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = marked_pp(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = marked_pp(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
