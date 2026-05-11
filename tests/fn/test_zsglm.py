"""Tests for morie.fn.zsglm -- Spatial GLMM simulation"""

import numpy as np
import pytest

from morie.fn.zsglm import spatial_glmm_sim


class TestSpatialGlmmSim:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_glmm_sim(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = spatial_glmm_sim(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
