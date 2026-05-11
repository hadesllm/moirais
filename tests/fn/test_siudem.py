"""Tests for morie.fn.siudem — SIU demographics."""

import pytest
import numpy as np
import pandas as pd
from morie.fn.siudem import siu_demographics
from morie.fn._containers import DescriptiveResult


class TestSiuDemographics:
    def test_basic(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"age": rng.integers(18, 65, 50), "gender": rng.choice(["M", "F"], 50)})
        r = siu_demographics(df)
        assert isinstance(r, DescriptiveResult)
        assert "mean_age" in r.extra

    def test_empty_df(self):
        df = pd.DataFrame({"age": [], "gender": []})
        r = siu_demographics(df)
        assert r.extra["n"] == 0
