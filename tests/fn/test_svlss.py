"""Tests for morie.fn.svlss -- Spatial loss function (quadratic/city-block)"""

import numpy as np
import pytest

from morie.fn.svlss import loss_function


class TestLossFunction:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = loss_function(data)
        assert result.value is not None

    def test_output_type(self):
        result = loss_function(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
