"""Tests for morie.fn.zerke -- Risk exceedance probability"""

import numpy as np
import pytest

from morie.fn.zerke import risk_exceedance


class TestRiskExceedance:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = risk_exceedance(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = risk_exceedance(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
