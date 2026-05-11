"""Test early_stopping."""
import numpy as np
from morie.fn.earlm import early_stopping, earlm
from morie.fn._containers import DescriptiveResult


class TestEarlyStopping:
    def test_should_not_stop(self):
        result = early_stopping([2.0, 1.5, 1.0], patience=5)
        assert result.value is False

    def test_should_stop(self):
        result = early_stopping([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6], patience=5)
        assert result.value is True

    def test_best_epoch(self):
        result = early_stopping([2.0, 1.0, 1.5, 1.5], patience=5)
        assert result.extra["best_epoch"] == 1
        assert abs(result.extra["best_loss"] - 1.0) < 1e-10

    def test_type(self):
        result = early_stopping([1.0, 0.9])
        assert isinstance(result, DescriptiveResult)

    def test_alias(self):
        assert earlm is early_stopping
