"""Tests for moirais.fn.zeklf -- Kulldorff spatial scan statistic"""

import numpy as np
import pytest

from moirais.fn.zeklf import kulldorff_scan


class TestKulldorffScan:
    def test_basic(self):
        observed = np.array([5, 3, 8, 2, 10])
        result = kulldorff_scan(observed)
        assert result.statistic is not None

    def test_output_type(self):
        result = kulldorff_scan(np.array([1,2,3,4,5]))
        assert hasattr(result, "statistic")
