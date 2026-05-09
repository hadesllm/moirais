"""Tests for moirais.fn.kreg -- Nadaraya-Watson kernel regression."""

import numpy as np
import pytest
from moirais.fn.kreg import kernel_regression, kreg
from moirais.fn._containers import DescriptiveResult


class TestKreg:
    def test_alias(self):
        assert kreg is kernel_regression

    def test_quadratic(self):
        x = np.linspace(-3, 3, 50)
        y = x ** 2
        x_pred = np.array([0.0, 1.0, 2.0])
        result = kernel_regression(x, y, x_pred, bandwidth=0.5)
        assert isinstance(result, DescriptiveResult)
        preds = result.value
        assert abs(preds[0] - 0.0) < 1.0
        assert abs(preds[1] - 1.0) < 1.0

    def test_defaults_to_train(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 4.0, 9.0])
        result = kernel_regression(x, y)
        assert len(result.value) == 3
