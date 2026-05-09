"""Tests for moirais.fn.sentl -- detection metrics."""

import numpy as np
from moirais.fn.sentl import detection_metrics, sentl
from moirais.fn._containers import DescriptiveResult


class TestSentl:
    def test_alias(self):
        assert sentl is detection_metrics

    def test_perfect(self):
        y_true = [1, 1, 0, 0]
        y_pred = [0.9, 0.8, 0.1, 0.2]
        result = detection_metrics(y_true, y_pred)
        assert isinstance(result, DescriptiveResult)
        assert result.value == 1.0

    def test_all_wrong(self):
        y_true = [1, 1, 0, 0]
        y_pred = [0.1, 0.2, 0.9, 0.8]
        result = detection_metrics(y_true, y_pred)
        assert result.value == 0.0
