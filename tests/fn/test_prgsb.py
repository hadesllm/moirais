"""Tests for moirais.fn.prgsb — program subgroup."""

import pytest
import numpy as np
import pandas as pd
from moirais.fn.prgsb import program_subgroup
from moirais.fn._containers import DescriptiveResult


class TestProgramSubgroup:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            "outcome": rng.normal(0, 1, n),
            "treatment": rng.binomial(1, 0.5, n),
            "subgroup": rng.choice(["A", "B"], n),
        })
        df.loc[df["treatment"] == 1, "outcome"] += 2.0
        r = program_subgroup(df)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["n_subgroups"] == 2

    def test_missing_col(self):
        with pytest.raises(ValueError):
            program_subgroup(pd.DataFrame({"x": [1]}))
