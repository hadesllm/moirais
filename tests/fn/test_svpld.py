"""Tests for morie.fn.svpld -- Dimensional polarization decomposition"""

import numpy as np
import pytest

from morie.fn.svpld import polarization_dim


class TestPolarizationDim:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = polarization_dim(data)
        assert result.value is not None

    def test_output_type(self):
        result = polarization_dim(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
