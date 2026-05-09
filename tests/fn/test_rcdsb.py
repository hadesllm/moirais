"""Tests for moirais.fn.rcdsb — recidivism subgroup."""

import pytest
import numpy as np
import pandas as pd
from moirais.fn.rcdsb import recidivism_subgroup
from moirais.fn._containers import DescriptiveResult


class TestRecidivismSubgroup:

    def test_returns_descriptive(self):
        df = pd.DataFrame({"recidivism": [1, 0, 1, 0, 1, 0], "group": ["A", "A", "B", "B", "C", "C"]})
        result = recidivism_subgroup(df)
        assert isinstance(result, DescriptiveResult)
        assert len(result.value) == 3

    def test_rates_bounded(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"recidivism": rng.integers(0, 2, 100), "group": rng.choice(["X", "Y", "Z"], 100)})
        result = recidivism_subgroup(df)
        assert (result.value["rate"] >= 0).all()
        assert (result.value["rate"] <= 1).all()
