"""Tests for morie.fn.zxfcm -- Fuzzy C-means spatial"""

import numpy as np
import pytest

from morie.fn.zxfcm import fuzzy_cmeans_sp


class TestFuzzyCmeansSp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = fuzzy_cmeans_sp(data)
        assert result.value is not None

    def test_output_type(self):
        result = fuzzy_cmeans_sp(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
