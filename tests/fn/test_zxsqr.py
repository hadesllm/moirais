"""Tests for moirais.fn.zxsqr -- Spatial quantile regression"""

import numpy as np
import pytest

from moirais.fn.zxsqr import spatial_quantile


class TestSpatialQuantile:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_quantile(data)
        assert result.value is not None

    def test_output_type(self):
        result = spatial_quantile(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
