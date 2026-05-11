"""Tests for morie.fn.dshot -- Precision at K."""

import numpy as np
from morie.fn.dshot import precision_at_k, dshot
from morie.fn._containers import DescriptiveResult


class TestDshot:
    def test_alias(self):
        assert dshot is precision_at_k

    def test_perfect(self):
        y_true = [1, 1, 1, 0, 0]
        scores = [0.9, 0.8, 0.7, 0.3, 0.1]
        result = precision_at_k(y_true, scores, k=3)
        assert isinstance(result, DescriptiveResult)
        assert result.value == 1.0

    def test_zero(self):
        y_true = [0, 0, 0, 1, 1]
        scores = [0.9, 0.8, 0.7, 0.3, 0.1]
        result = precision_at_k(y_true, scores, k=3)
        assert result.value == 0.0
