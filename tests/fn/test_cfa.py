"""Tests for morie.fn.cfa — Confirmatory Factor Analysis."""

import numpy as np
import pandas as pd
import pytest
from morie.fn.cfa import cfa
from morie.fn._containers import CfaRes


def _make_two_factor_data(rng, n=300):
    """Generate data with clear 2-factor structure."""
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    df = pd.DataFrame({
        "a1": f1 + rng.standard_normal(n) * 0.3,
        "a2": f1 + rng.standard_normal(n) * 0.3,
        "a3": f1 + rng.standard_normal(n) * 0.3,
        "b1": f2 + rng.standard_normal(n) * 0.3,
        "b2": f2 + rng.standard_normal(n) * 0.3,
        "b3": f2 + rng.standard_normal(n) * 0.3,
    })
    return df


class TestCfa:
    """Tests for Confirmatory Factor Analysis."""

    def test_returns_cfa_res(self, rng):
        df = _make_two_factor_data(rng)
        structure = {"F1": ["a1", "a2", "a3"], "F2": ["b1", "b2", "b3"]}
        result = cfa(df, structure)
        assert isinstance(result, CfaRes)

    def test_fit_indices_range(self, rng):
        """Good-fitting model should have reasonable fit indices."""
        df = _make_two_factor_data(rng)
        structure = {"F1": ["a1", "a2", "a3"], "F2": ["b1", "b2", "b3"]}
        result = cfa(df, structure)
        assert 0 <= result.cfi <= 1
        assert 0 <= result.tli <= 1
        assert result.rmsea >= 0
        assert result.srmr >= 0

    def test_loadings_populated(self, rng):
        df = _make_two_factor_data(rng)
        structure = {"F1": ["a1", "a2", "a3"], "F2": ["b1", "b2", "b3"]}
        result = cfa(df, structure)
        assert "F1" in result.loadings
        assert len(result.loadings["F1"]) == 3
        # Loadings should be positive for well-structured data
        assert all(v > 0 for v in result.loadings["F1"].values())

    def test_residuals_shape(self, rng):
        df = _make_two_factor_data(rng)
        structure = {"F1": ["a1", "a2", "a3"], "F2": ["b1", "b2", "b3"]}
        result = cfa(df, structure)
        assert result.residuals.shape == (6, 6)
