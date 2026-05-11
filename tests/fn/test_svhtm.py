"""Tests for morie.fn.svhtm -- Multi-candidate Hotelling model"""

import numpy as np
import pytest

from morie.fn.svhtm import hotelling_multi


class TestHotellingMulti:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = hotelling_multi(data)
        assert result.value is not None

    def test_output_type(self):
        result = hotelling_multi(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
