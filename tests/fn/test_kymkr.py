"""Tests for moirais.fn.kymkr -- information gain."""

import numpy as np
import pandas as pd
from moirais.fn.kymkr import information_gain, kymkr
from moirais.fn._containers import DescriptiveResult


class TestKymkr:
    def test_alias(self):
        assert kymkr is information_gain

    def test_perfect_predictor(self):
        df = pd.DataFrame({"outcome": [0, 0, 1, 1], "x": [0, 0, 1, 1]})
        result = information_gain(df, target="outcome", feature="x")
        assert isinstance(result, DescriptiveResult)
        assert result.value > 0.5

    def test_random_feature(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"outcome": rng.integers(0, 2, 200), "x": rng.normal(0, 1, 200)})
        result = information_gain(df, target="outcome", feature="x")
        assert result.value < 0.5
