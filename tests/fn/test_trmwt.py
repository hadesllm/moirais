"""Tests for moirais.fn.trmwt -- Trimmed weights."""

import numpy as np
import pandas as pd
import pytest
from moirais.fn.trmwt import trimmed_weights, trmwt


class TestTrimmedWeights:
    def test_alias(self):
        assert trmwt is trimmed_weights

    def test_quantile_trimming(self):
        rng = np.random.default_rng(42)
        weights = np.concatenate([np.ones(90), np.array([100.0] * 10)])
        df = pd.DataFrame({"weight": weights})
        trimmed = trimmed_weights(df, lower=0.01, upper=0.90)
        assert trimmed.max() < 100.0

    def test_fixed_trimming(self):
        df = pd.DataFrame({"weight": [0.01, 1.0, 50.0, 100.0]})
        trimmed = trimmed_weights(df, method="fixed", threshold=10.0)
        assert trimmed.max() <= 10.0
        assert trimmed.min() >= 0.1

    def test_crump_requires_ps(self):
        df = pd.DataFrame({"weight": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="ps"):
            trimmed_weights(df, method="crump")

    def test_crump_zeros_extreme(self):
        rng = np.random.default_rng(42)
        n = 200
        ps = np.concatenate([rng.uniform(0.001, 0.01, 20),
                             rng.uniform(0.2, 0.8, 160),
                             rng.uniform(0.99, 0.999, 20)])
        w = np.ones(n)
        df = pd.DataFrame({"weight": w, "ps": ps})
        trimmed = trimmed_weights(df, method="crump")
        assert (trimmed == 0).sum() > 0
