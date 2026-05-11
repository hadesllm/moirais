"""Tests for morie.fn.zesmr -- Standardized Morbidity Ratio"""

import numpy as np
import pytest

from morie.fn.zesmr import smr_compute


class TestSmrCompute:
    def test_basic(self):
        observed = np.array([5, 3, 8, 2, 10])
        result = smr_compute(observed)
        assert result.statistic is not None

    def test_output_type(self):
        result = smr_compute(np.array([1,2,3,4,5]))
        assert hasattr(result, "statistic")
