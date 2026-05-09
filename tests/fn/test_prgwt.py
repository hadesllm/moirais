"""Tests for moirais.fn.prgwt — program waitlist IV."""

import pytest
import numpy as np
import pandas as pd
from moirais.fn.prgwt import program_waitlist
from moirais.fn._containers import ESRes


class TestProgramWaitlist:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 500
        z = rng.binomial(1, 0.5, n).astype(float)
        d = (z + rng.normal(0, 0.5, n) > 0.5).astype(float)
        y = 3.0 * d + rng.normal(0, 1, n)
        df = pd.DataFrame({"waitlisted": z, "enrolled": d, "outcome": y})
        r = program_waitlist(df)
        assert isinstance(r, ESRes)
        assert r.estimate == pytest.approx(3.0, abs=1.5)

    def test_missing_col(self):
        with pytest.raises(ValueError):
            program_waitlist(pd.DataFrame({"x": [1]}))
