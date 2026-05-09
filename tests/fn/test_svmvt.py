"""Tests for moirais.fn.svmvt -- Median voter theorem test"""

import numpy as np
import pytest

from moirais.fn.svmvt import median_voter


class TestMedianVoter:
    def test_basic(self):
        x = np.array([1.0, 2.0])
        result = median_voter(x, ideal_point=np.array([0.0, 0.0]))
        assert result.value is not None
        assert result.value >= 0

    def test_output_type(self):
        result = median_voter(np.array([1.0, 2.0]))
        assert hasattr(result, "value")
