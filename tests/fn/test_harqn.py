"""Tests for moirais.fn.harqn -- Harrell's C concordance."""

import numpy as np
import pandas as pd
from moirais.fn.harqn import harrells_c, harqn
from moirais.fn._containers import DescriptiveResult


class TestHarqn:
    def test_alias(self):
        assert harqn is harrells_c

    def test_perfect_concordance(self):
        df = pd.DataFrame({
            "time": [1, 2, 3, 4, 5],
            "event": [1, 1, 1, 1, 1],
            "risk_score": [5, 4, 3, 2, 1],
        })
        result = harrells_c(df)
        assert isinstance(result, DescriptiveResult)
        assert result.value == 1.0

    def test_random(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "time": rng.exponential(1, 30),
            "event": rng.binomial(1, 0.7, 30),
            "risk_score": rng.normal(0, 1, 30),
        })
        result = harrells_c(df)
        assert 0 <= result.value <= 1
