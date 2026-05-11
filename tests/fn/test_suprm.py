"""Tests for morie.fn.suprm -- KS supremum test."""

import numpy as np
import pandas as pd
from morie.fn.suprm import ks_supremum, suprm
from morie.fn._containers import TestResult


class TestSuprm:
    def test_alias(self):
        assert suprm is ks_supremum

    def test_normal_data(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.normal(0, 1, 200)})
        result = ks_supremum(df, dist="norm")
        assert isinstance(result, TestResult)
        assert result.p_value > 0.05

    def test_non_normal(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.exponential(1, 200)})
        result = ks_supremum(df, dist="norm")
        assert result.p_value < 0.05
