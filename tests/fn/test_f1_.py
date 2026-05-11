"""Tests for morie.fn.f1_ -- F1 score."""

import numpy as np
import pytest
from morie.fn.f1_ import f1_score


class TestF1Score:
    def test_perfect(self):
        y = np.array([0, 0, 1, 1])
        result = f1_score(y, y)
        assert result["f1"] == pytest.approx(1.0, abs=1e-10)
        assert result["precision"] == pytest.approx(1.0, abs=1e-10)
        assert result["recall"] == pytest.approx(1.0, abs=1e-10)

    def test_no_positives_predicted(self):
        result = f1_score([1, 1, 1], [0, 0, 0])
        assert result["recall"] == pytest.approx(0.0, abs=1e-10)

    def test_support_count(self):
        result = f1_score([0, 0, 1, 1, 1], [0, 0, 1, 0, 1])
        assert result["support"] == 3
