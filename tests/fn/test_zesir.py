"""Tests for morie.fn.zesir -- Spatial SIR diffusion"""

import numpy as np
import pytest

from morie.fn.zesir import spatial_sir


class TestSpatialSir:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_sir(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = spatial_sir(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
