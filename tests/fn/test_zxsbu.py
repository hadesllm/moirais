"""Tests for moirais.fn.zxsbu -- Buffered spatial CV"""

import numpy as np
import pytest

from moirais.fn.zxsbu import spatial_cv_buffer


class TestSpatialCvBuffer:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_cv_buffer(data)
        assert result.value is not None

    def test_output_type(self):
        result = spatial_cv_buffer(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
