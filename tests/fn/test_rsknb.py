"""Tests for morie.fn.rsknb — risk Nagelkerke R2."""

import pytest
import numpy as np
from morie.fn.rsknb import risk_nagelkerke
from morie.fn._containers import ESRes


class TestRiskNagelkerke:

    def test_returns_esres(self):
        rng = np.random.default_rng(42)
        outcomes = rng.integers(0, 2, 100).astype(float)
        probs = np.clip(outcomes + rng.standard_normal(100) * 0.2, 0.01, 0.99)
        result = risk_nagelkerke(probs, outcomes)
        assert isinstance(result, ESRes)

    def test_r2_bounded(self):
        rng = np.random.default_rng(7)
        outcomes = rng.integers(0, 2, 50).astype(float)
        probs = np.clip(outcomes * 0.8 + 0.1, 0.01, 0.99)
        result = risk_nagelkerke(probs, outcomes)
        assert 0 <= result.estimate <= 1
