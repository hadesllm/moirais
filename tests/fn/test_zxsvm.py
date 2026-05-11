"""Tests for morie.fn.zxsvm -- Spatial SVM"""

import numpy as np
import pytest

from morie.fn.zxsvm import spatial_svm


class TestSpatialSvm:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_svm(data)
        assert result.value is not None

    def test_output_type(self):
        result = spatial_svm(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
