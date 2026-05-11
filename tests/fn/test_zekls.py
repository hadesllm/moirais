"""Tests for morie.fn.zekls -- Circular scan statistic"""

import numpy as np
import pytest

from morie.fn.zekls import scan_circular


class TestScanCircular:
    def test_basic(self):
        observed = np.array([5, 3, 8, 2, 10])
        result = scan_circular(observed)
        assert result.statistic is not None

    def test_output_type(self):
        result = scan_circular(np.array([1,2,3,4,5]))
        assert hasattr(result, "statistic")
