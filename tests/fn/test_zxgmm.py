"""Tests for moirais.fn.zxgmm -- Spatial Gaussian mixture"""

import numpy as np
import pytest

from moirais.fn.zxgmm import gmm_spatial


class TestGmmSpatial:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gmm_spatial(data)
        assert result.value is not None

    def test_output_type(self):
        result = gmm_spatial(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
