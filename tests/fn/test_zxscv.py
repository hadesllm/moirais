"""Tests for moirais.fn.zxscv -- Spatial LOO cross-validation"""

import numpy as np
import pytest

from moirais.fn.zxscv import spatial_cv_loo


class TestSpatialCvLoo:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_cv_loo(data)
        assert result.value is not None

    def test_output_type(self):
        result = spatial_cv_loo(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
