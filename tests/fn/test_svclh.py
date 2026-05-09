"""Tests for moirais.fn.svclh -- Heart of spatial game"""

import numpy as np
import pytest

from moirais.fn.svclh import coalition_heart


class TestCoalitionHeart:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = coalition_heart(data)
        assert result.value is not None

    def test_output_type(self):
        result = coalition_heart(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
