"""Tests for morie.fn.zxsls -- Spatial LASSO"""

import numpy as np
import pytest

from morie.fn.zxsls import spatial_lasso


class TestSpatialLasso:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_lasso(data)
        assert result.value is not None

    def test_output_type(self):
        result = spatial_lasso(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
