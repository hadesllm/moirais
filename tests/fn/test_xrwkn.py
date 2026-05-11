"""Tests for morie.fn.xrwkn -- K-nearest neighbors weights"""

import numpy as np
import pytest

from morie.fn.xrwkn import w_knn


class TestWKnn:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = w_knn(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = w_knn(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
