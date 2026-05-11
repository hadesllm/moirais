"""Tests for morie.fn.bane -- Structural break test."""

import numpy as np
import pandas as pd
from morie.fn.bane import structural_break, bane
from morie.fn._containers import TestResult


class TestBane:
    def test_alias(self):
        assert bane is structural_break

    def test_detects_break(self):
        x = np.concatenate([np.zeros(50), np.ones(50) * 10])
        df = pd.DataFrame({"x": x})
        result = structural_break(df, col="x")
        assert isinstance(result, TestResult)
        assert result.p_value < 0.05
        assert abs(result.extra["breakpoint"] - 50) < 15

    def test_no_break(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.normal(0, 1, 100)})
        result = structural_break(df, col="x")
        assert result.p_value > 0.01
