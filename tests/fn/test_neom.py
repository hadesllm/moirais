"""Tests for moirais.fn.neom -- decision tree split criterion."""

import numpy as np
import pandas as pd
from moirais.fn.neom import decision_split, neom
from moirais.fn._containers import DescriptiveResult


class TestNeom:
    def test_alias(self):
        assert neom is decision_split

    def test_gini_split(self):
        rng = np.random.default_rng(42)
        x = np.concatenate([rng.normal(0, 1, 50), rng.normal(3, 1, 50)])
        y = np.array([0] * 50 + [1] * 50)
        df = pd.DataFrame({"outcome": y, "x": x})
        result = decision_split(df, target="outcome", feature="x")
        assert isinstance(result, DescriptiveResult)
        assert result.extra["criterion"] == "gini"
        assert 0.5 < result.value < 2.5

    def test_entropy(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        df = pd.DataFrame({"outcome": y, "x": x})
        result = decision_split(df, target="outcome", feature="x", criterion="entropy")
        assert result.extra["criterion"] == "entropy"
        assert result.extra["impurity_reduction"] > 0
