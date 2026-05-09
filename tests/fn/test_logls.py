"""Tests for moirais.fn.logls -- Log loss."""

import numpy as np
import pytest
from moirais.fn.logls import log_loss


class TestLogLoss:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1], dtype=float)
        y_prob = np.array([0.001, 0.001, 0.999, 0.999])
        loss = log_loss(y_true, y_prob)
        assert loss < 0.01

    def test_worst_predictions_high(self):
        y_true = np.array([0, 0, 1, 1], dtype=float)
        y_prob = np.array([0.999, 0.999, 0.001, 0.001])
        loss = log_loss(y_true, y_prob)
        assert loss > 5.0

    def test_uniform_predictions(self):
        y_true = np.array([0, 1], dtype=float)
        y_prob = np.array([0.5, 0.5])
        loss = log_loss(y_true, y_prob)
        assert loss == pytest.approx(np.log(2), abs=1e-10)
