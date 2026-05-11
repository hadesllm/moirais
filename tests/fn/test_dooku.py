"""Tests for morie.fn.dooku -- Double ML alias (Count Dooku)."""

import pytest


class TestDooku:
    def test_is_callable(self):
        """dooku should be a callable alias for estimate_double_ml."""
        from morie.fn.dooku import double_ml
        assert callable(double_ml)

    def test_runs_on_binary_df(self, binary_df):
        """Double ML should run and return a fitted DoubleMLPLR object."""
        from morie.fn.dooku import double_ml
        result = double_ml(
            data=binary_df,
            outcome="outcome",
            treatment="treatment",
            covariates=["x1", "x2"],
        )
        # Returns a DoubleMLPLR object
        assert hasattr(result, "coef")
        assert hasattr(result, "se")
        assert hasattr(result, "pval")

    def test_ate_near_true(self, binary_df):
        """ATE estimate should be in a reasonable range (true effect ~ 0.5)."""
        from morie.fn.dooku import double_ml
        result = double_ml(
            data=binary_df,
            outcome="outcome",
            treatment="treatment",
            covariates=["x1", "x2"],
        )
        ate = float(result.coef[0])
        assert -1.0 < ate < 2.0  # reasonable range for true=0.5
