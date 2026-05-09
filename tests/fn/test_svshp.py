"""Tests for moirais.fn.svshp -- Shapley value in spatial game"""

import numpy as np
import pytest

from moirais.fn.svshp import shapley_spatial


class TestShapleySpatial:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = shapley_spatial(data)
        assert result.value is not None

    def test_output_type(self):
        result = shapley_spatial(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
