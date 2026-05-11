"""Tests for morie.fn.zerrk -- Relative risk kernel ratio"""

import numpy as np
import pytest

from morie.fn.zerrk import relative_risk_kern


class TestRelativeRiskKern:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = relative_risk_kern(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = relative_risk_kern(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
