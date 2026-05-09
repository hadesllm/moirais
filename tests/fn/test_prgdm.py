"""Tests for moirais.fn.prgdm — program DML."""

import pytest
import numpy as np
import pandas as pd
from moirais.fn.prgdm import program_dml
from moirais.fn._containers import ESRes


class TestProgramDml:
    def test_positive_effect(self):
        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        d = (rng.normal(x, 1) > 0).astype(float)
        y = 2.0 * d + x + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": d, "x1": x})
        r = program_dml(df)
        assert isinstance(r, ESRes)
        assert r.estimate == pytest.approx(2.0, abs=0.5)

    def test_missing_col(self):
        with pytest.raises(ValueError):
            program_dml(pd.DataFrame({"x": [1]}))
