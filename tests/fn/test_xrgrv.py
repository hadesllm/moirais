"""Tests for morie.fn.xrgrv -- Gravity spatial interaction"""

import numpy as np
import pytest

from morie.fn.xrgrv import gravity_model


class TestGravityModel:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gravity_model(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = gravity_model(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
