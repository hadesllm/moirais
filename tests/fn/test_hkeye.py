"""Tests for moirais.fn.hkeye -- precision-recall at top-k."""

import numpy as np
from moirais.fn.hkeye import precision_recall_at_k, hkeye
from moirais.fn._containers import DescriptiveResult


class TestHkeye:
    def test_alias(self):
        assert hkeye is precision_recall_at_k

    def test_perfect_topk(self):
        y_true = np.array([1, 1, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        r = precision_recall_at_k(y_true, scores, k=3)
        assert isinstance(r, DescriptiveResult)
        assert r.value["precision_at_k"] == 1.0
        assert r.value["recall_at_k"] == 1.0

    def test_partial_precision(self):
        y_true = np.array([1, 0, 0, 0, 1])
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        r = precision_recall_at_k(y_true, scores, k=2)
        assert r.value["precision_at_k"] == 0.5
