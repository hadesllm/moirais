"""Tests for morie.fn.zeklr -- Retrospective space-time scan"""

import numpy as np
import pytest

from morie.fn.zeklr import scan_retrospective


class TestScanRetrospective:
    def test_basic(self):
        observed = np.array([5, 3, 8, 2, 10])
        result = scan_retrospective(observed)
        assert result.statistic is not None

    def test_output_type(self):
        result = scan_retrospective(np.array([1,2,3,4,5]))
        assert hasattr(result, "statistic")
