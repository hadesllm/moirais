"""Tests for morie.fn.zerad -- Radiation model (mobility)"""

import numpy as np
import pytest

from morie.fn.zerad import radiation_model


class TestRadiationModel:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = radiation_model(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = radiation_model(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
