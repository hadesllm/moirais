"""Tests for morie.fn.osyn1 — OTIS synthetic control."""

import pytest
import numpy as np
import pandas as pd
from morie.fn.osyn1 import otis_synth_region
from morie.fn._containers import DescriptiveResult


class TestOtisSynthRegion:

    def test_returns_descriptive(self):
        rng = np.random.default_rng(42)
        rows = []
        for r in ["treated", "A", "B", "C"]:
            for p in range(1, 11):
                rows.append({"region": r, "period": p, "outcome": rng.standard_normal() + (2 if r == "treated" and p > 5 else 0)})
        df = pd.DataFrame(rows)
        result = otis_synth_region(df, treated_region="treated")
        assert isinstance(result, DescriptiveResult)

    def test_weights_nonnegative(self):
        rng = np.random.default_rng(7)
        rows = []
        for r in ["T", "D1", "D2"]:
            for p in range(1, 9):
                rows.append({"region": r, "period": p, "outcome": rng.standard_normal()})
        df = pd.DataFrame(rows)
        result = otis_synth_region(df, treated_region="T")
        for v in result.extra["weights"].values():
            assert v >= 0
