"""Tests for moirais.fn.svdsc -- Discounted directional utility"""

import numpy as np
import pytest

from moirais.fn.svdsc import discount_utility


class TestDiscountUtility:
    def test_basic(self):
        x = np.array([1.0, 2.0])
        result = discount_utility(x, ideal_point=np.array([0.0, 0.0]))
        assert result.value is not None
        assert result.value >= 0

    def test_output_type(self):
        result = discount_utility(np.array([1.0, 2.0]))
        assert hasattr(result, "value")
