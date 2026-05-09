"""Tests for moirais.fn.svmv2 -- Median voter in 2D (Plott conditions)"""

import numpy as np
import pytest

from moirais.fn.svmv2 import median_voter_2d


class TestMedianVoter2d:
    def test_basic(self):
        x = np.array([1.0, 2.0])
        result = median_voter_2d(x, ideal_point=np.array([0.0, 0.0]))
        assert result.value is not None
        assert result.value >= 0

    def test_output_type(self):
        result = median_voter_2d(np.array([1.0, 2.0]))
        assert hasattr(result, "value")
