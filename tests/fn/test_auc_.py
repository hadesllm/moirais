"""Tests for morie.fn.auc_ -- AUC score."""

import numpy as np
import pytest
from morie.fn.auc_ import auc_score


class TestAucScore:
    def test_perfect_auc(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        auc = auc_score(y_true, y_score)
        assert auc == pytest.approx(1.0, abs=0.01)

    def test_random_auc(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=1000)
        y_score = rng.random(1000)
        auc = auc_score(y_true, y_score)
        assert 0.4 < auc < 0.6  # should be near 0.5

    def test_single_class_raises(self):
        with pytest.raises(ValueError, match="both classes"):
            auc_score(np.ones(5), np.ones(5))
