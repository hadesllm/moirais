"""Tests for morie.fn.zeitv -- Travel time catchment"""

import numpy as np
import pytest

from morie.fn.zeitv import travel_time_catch


class TestTravelTimeCatch:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = travel_time_catch(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = travel_time_catch(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
