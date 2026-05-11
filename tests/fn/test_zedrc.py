"""Tests for morie.fn.zedrc -- Spatial dose-response curve"""

import numpy as np
import pytest

from morie.fn.zedrc import dose_resp_spatial


class TestDoseRespSpatial:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dose_resp_spatial(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = dose_resp_spatial(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
