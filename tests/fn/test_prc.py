"""Tests for morie.fn.prc — Precision-recall curve."""

import numpy as np
import pytest

from morie.fn.prc import pr_curve


class TestPRCurve:
    def test_perfect_classifier(self):
        y = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        res = pr_curve(y, scores)
        assert res.extra["auprc"] > 0.5

    def test_random_classifier(self):
        rng = np.random.default_rng(42)
        y = rng.choice([0, 1], 200)
        scores = rng.uniform(0, 1, 200)
        res = pr_curve(y, scores)
        assert 0 < res.extra["auprc"] < 1

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            pr_curve([1, 0], [0.5])
