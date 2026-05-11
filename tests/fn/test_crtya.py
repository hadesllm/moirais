"""Tests for morie.fn.crtya — youth court."""

import pytest
import numpy as np
import pandas as pd
from morie.fn.crtya import court_youth
from morie.fn._containers import DescriptiveResult


class TestYouthCourt:
    def test_basic(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "age": rng.integers(12, 18, 50),
            "sentence_type": rng.choice(["Probation", "Custodial", "Community"], 50),
            "days_to_disposition": rng.integers(30, 300, 50),
        })
        r = court_youth(df)
        assert isinstance(r, DescriptiveResult)
        assert "pct_custodial" in r.extra

    def test_empty(self):
        df = pd.DataFrame({"age": []})
        r = court_youth(df)
        assert r.extra["n"] == 0
