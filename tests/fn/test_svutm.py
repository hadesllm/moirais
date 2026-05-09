"""Tests for moirais.fn.svutm -- Spatial utility maximizer"""

import numpy as np
import pytest

from moirais.fn.svutm import utility_max


class TestUtilityMax:
    def test_basic(self):
        x = np.array([1.0, 2.0])
        result = utility_max(x, ideal_point=np.array([0.0, 0.0]))
        assert result.value is not None
        assert result.value >= 0

    def test_output_type(self):
        result = utility_max(np.array([1.0, 2.0]))
        assert hasattr(result, "value")
