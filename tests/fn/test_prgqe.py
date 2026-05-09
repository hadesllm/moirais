"""Tests for moirais.fn.prgqe — program quasi-experimental."""

import pytest
import numpy as np
import pandas as pd
from moirais.fn.prgqe import program_quasi_exp
from moirais.fn._containers import ESRes


class TestProgramQuasiExp:
    def test_positive_did(self):
        rng = np.random.default_rng(42)
        n = 400
        d = np.repeat([0, 1], n // 2)
        t = np.tile([0, 1], n // 2)
        y = 5.0 + 1.0 * d + 2.0 * t + 3.0 * d * t + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": d, "post": t})
        r = program_quasi_exp(df)
        assert isinstance(r, ESRes)
        assert r.estimate == pytest.approx(3.0, abs=1.0)

    def test_missing_col(self):
        with pytest.raises(ValueError):
            program_quasi_exp(pd.DataFrame({"x": [1]}))
