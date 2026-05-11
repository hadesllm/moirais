"""Tests for morie.fn.wolvn -- MICE imputation."""

import numpy as np
import pandas as pd
from morie.fn.wolvn import mice_impute, wolvn
from morie.fn._containers import DescriptiveResult


class TestWolvn:
    def test_alias(self):
        assert wolvn is mice_impute

    def test_basic_imputation(self):
        df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [10, np.nan, 30, 40]})
        r = mice_impute(df, n_imputations=3, n_iter=5, seed=42)
        assert isinstance(r, DescriptiveResult)
        assert len(r.value["imputed"]) == 3
        for imp_df in r.value["imputed"]:
            assert imp_df.isna().sum().sum() == 0

    def test_no_missing(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        r = mice_impute(df)
        assert r.value["frac_missing"] == 0.0
