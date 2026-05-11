"""Tests for morie.fn.xrsar -- SAR (Spatial Lag) model ML estimation"""

import numpy as np
import pytest

from morie.fn.xrsar import sar_ml


class TestSarMl:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sar_ml(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = sar_ml(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
