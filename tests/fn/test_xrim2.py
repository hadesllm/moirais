"""Tests for morie.fn.xrim2 -- SDM impacts decomposition"""

import numpy as np
import pytest

from morie.fn.xrim2 import sdm_impacts


class TestSdmImpacts:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sdm_impacts(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = sdm_impacts(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
