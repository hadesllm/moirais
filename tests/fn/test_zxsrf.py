"""Tests for moirais.fn.zxsrf -- Spatial random forest"""

import numpy as np
import pytest

from moirais.fn.zxsrf import spatial_rf


class TestSpatialRf:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_rf(data)
        assert result.value is not None

    def test_output_type(self):
        result = spatial_rf(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
