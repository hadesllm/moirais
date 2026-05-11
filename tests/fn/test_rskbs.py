"""Tests for morie.fn.rskbs — risk Brier score."""

import pytest
import numpy as np
from morie.fn.rskbs import risk_brier
from morie.fn._containers import ESRes


class TestRiskBrier:

    def test_returns_esres(self):
        probs = np.array([0.1, 0.9, 0.3, 0.8])
        outcomes = np.array([0, 1, 0, 1])
        result = risk_brier(probs, outcomes)
        assert isinstance(result, ESRes)
        assert result.estimate >= 0

    def test_perfect_score_zero(self):
        probs = np.array([0.0, 1.0, 0.0, 1.0])
        outcomes = np.array([0, 1, 0, 1])
        result = risk_brier(probs, outcomes)
        assert result.estimate == pytest.approx(0.0)
