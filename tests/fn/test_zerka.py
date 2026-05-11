"""Tests for morie.fn.zerka -- Adaptive bandwidth relative risk"""

import numpy as np
import pytest

from morie.fn.zerka import risk_adaptive_bw


class TestRiskAdaptiveBw:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = risk_adaptive_bw(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = risk_adaptive_bw(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
