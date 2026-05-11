"""Tests for morie.fn.zebuf -- Buffer-based exposure assessment"""

import numpy as np
import pytest

from morie.fn.zebuf import buffer_exposure


class TestBufferExposure:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = buffer_exposure(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = buffer_exposure(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
