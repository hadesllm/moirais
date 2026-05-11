"""Tests for morie.fn.zxsbv -- Spatial block cross-validation"""

import numpy as np
import pytest

from morie.fn.zxsbv import spatial_cv_block


class TestSpatialCvBlock:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_cv_block(data)
        assert result.value is not None

    def test_output_type(self):
        result = spatial_cv_block(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
