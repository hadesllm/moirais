"""Tests for moirais.fn.mtodr — driver risk."""

import pytest
import numpy as np
import pandas as pd
from moirais.fn.mtodr import mto_driver_risk
from moirais.fn._containers import DescriptiveResult


class TestDriverRisk:
    def test_basic(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "age_group": rng.choice(["16-24", "25-44", "45+"], 200),
            "gender": rng.choice(["M", "F"], 200),
            "crash": rng.integers(0, 2, 200),
        })
        r = mto_driver_risk(df)
        assert isinstance(r, DescriptiveResult)

    def test_missing_col(self):
        with pytest.raises(ValueError):
            mto_driver_risk(pd.DataFrame({"x": [1]}))
