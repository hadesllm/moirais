"""Tests for moirais.fn.zxvnc -- Vincenty geodesic distance"""

import numpy as np
import pytest

from moirais.fn.zxvnc import vincenty_dist


class TestVincentyDist:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = vincenty_dist(data)
        assert result.value is not None

    def test_output_type(self):
        result = vincenty_dist(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
