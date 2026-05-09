"""Tests for moirais.fn.zxhvr -- Haversine distance"""

import numpy as np
import pytest

from moirais.fn.zxhvr import haversine_dist


class TestHaversineDist:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = haversine_dist(data)
        assert result.value is not None

    def test_output_type(self):
        result = haversine_dist(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
