"""Tests for morie.fn.nmplr -- Legislative polarization"""

import numpy as np
import pytest

from morie.fn.nmplr import leg_polarize


class TestLegPolarize:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = leg_polarize(data)
        assert result.value is not None

    def test_output_type(self):
        result = leg_polarize(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
