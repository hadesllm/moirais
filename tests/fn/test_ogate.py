"""Tests for moirais.fn.ogate — GATE by age group."""

import pandas as pd
from moirais.fn.ogate import otis_gate_age


class TestOtisGateAge:
    def test_returns_dataframe(self, otis_df):
        result = otis_gate_age(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_age_groups(self, otis_df):
        result = otis_gate_age(otis_df)
        assert len(result) == otis_df["age_group"].nunique()

    def test_columns(self, otis_df):
        result = otis_gate_age(otis_df)
        for col in ("age_group", "gate", "se", "pval", "n1", "n0"):
            assert col in result.columns

    def test_pval_range(self, otis_df):
        result = otis_gate_age(otis_df)
        assert (result["pval"] >= 0).all()
        assert (result["pval"] <= 1).all()
