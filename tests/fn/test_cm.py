"""Tests for morie.fn.cm -- Confusion matrix."""

import numpy as np
import pytest
from morie.fn.cm import confusion_matrix


class TestConfusionMatrix:
    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 1])
        result = confusion_matrix(y, y)
        assert result["tp"] == 2
        assert result["tn"] == 2
        assert result["fp"] == 0
        assert result["fn"] == 0
        assert result["accuracy"] == 1.0
        assert result["f1"] == 1.0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        result = confusion_matrix(y_true, y_pred)
        assert result["tp"] == 0
        assert result["tn"] == 0
        assert result["accuracy"] == 0.0

    def test_matrix_shape(self):
        result = confusion_matrix([0, 1, 0, 1], [0, 0, 1, 1])
        assert result["matrix"].shape == (2, 2)
