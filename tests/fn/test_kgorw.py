"""Tests for morie.fn.kgorw -- Ordinary kriging weights"""

import numpy as np
import pytest

from morie.fn.kgorw import ok_weights


class TestOkWeights:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ok_weights(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = ok_weights(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
