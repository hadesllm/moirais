"""Tests for morie.fn.svsph -- Spatial phase transition (chaos/order)"""

import numpy as np
import pytest

from morie.fn.svsph import spatial_phase


class TestSpatialPhase:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_phase(data)
        assert result.value is not None

    def test_output_type(self):
        result = spatial_phase(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
