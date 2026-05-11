"""Tests for morie.fn.kgckw -- Co-kriging weights"""

import numpy as np
import pytest

from morie.fn.kgckw import cok_weights


class TestCokWeights:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = cok_weights(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = cok_weights(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
