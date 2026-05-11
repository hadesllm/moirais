"""Tests for morie.fn.zeklp -- Prospective space-time scan"""

import numpy as np
import pytest

from morie.fn.zeklp import scan_prospective


class TestScanProspective:
    def test_basic(self):
        observed = np.array([5, 3, 8, 2, 10])
        result = scan_prospective(observed)
        assert result.statistic is not None

    def test_output_type(self):
        result = scan_prospective(np.array([1,2,3,4,5]))
        assert hasattr(result, "statistic")
