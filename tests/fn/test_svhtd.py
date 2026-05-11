"""Tests for morie.fn.svhtd -- Hotelling-Downs convergence equilibrium"""

import numpy as np
import pytest

from morie.fn.svhtd import hotelling_downs


class TestHotellingDowns:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = hotelling_downs(data)
        assert result.value is not None

    def test_output_type(self):
        result = hotelling_downs(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
