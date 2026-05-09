"""Tests for moirais.otis — Ontario Restrictive Confinement analysis."""

import numpy as np
import pandas as pd
import pytest

from moirais.otis import (
    RplRes, AstRes, VolRes, OtDmlR,
    rplace, astcmb, volat, rctrnd, otdesc, otdml,
    REGIONS, AGE_GROUPS, ALERT_COMBOS,
)


@pytest.fixture
def otis_df():
    """Synthetic OTIS-like correctional placement data."""
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame({
        "unique_individual_id": [f"P{i:04d}" for i in rng.choice(100, n)],
        "end_fiscal_year": rng.choice([2023, 2024, 2025], n),
        "region_at_time_of_placement": rng.choice(REGIONS, n),
        "region_most_recent_placement": rng.choice(REGIONS, n),
        "gender": rng.choice(["Male", "Female"], n),
        "age_category": rng.choice(AGE_GROUPS, n),
        "mental_health_alert": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "suicide_risk_alert": rng.choice(["Yes", "No"], n, p=[0.15, 0.85]),
        "suicide_watch_alert": rng.choice(["Yes", "No"], n, p=[0.05, 0.95]),
        "number_of_placements": rng.integers(1, 10, n),
        "Y": rng.choice([0, 1], n, p=[0.85, 0.15]),
        "D": rng.choice([0, 1], n, p=[0.7, 0.3]),
    })


class TestConstants:
    def test_regions(self):
        assert len(REGIONS) == 5
        assert "Toronto" in REGIONS

    def test_age_groups(self):
        assert len(AGE_GROUPS) == 3

    def test_alert_combos(self):
        assert len(ALERT_COMBOS) == 8
        assert ALERT_COMBOS["a7"] == (1, 1, 1)
        assert ALERT_COMBOS["a8"] == (0, 0, 0)


class TestRplace:
    def test_returns_rplres(self, otis_df):
        r = rplace(otis_df, year=2023)
        assert isinstance(r, RplRes)
        assert r.year == 2023

    def test_props_sum_to_one(self, otis_df):
        r = rplace(otis_df, year=2024)
        row_sums = r.props.sum(axis=1)
        for s in row_sums:
            assert abs(s - 1.0) < 0.01 or s == 0

    def test_sex_filter(self, otis_df):
        r = rplace(otis_df, year=2023, sex="Male")
        assert r.sex == "Male"

    def test_all_regions_in_columns(self, otis_df):
        r = rplace(otis_df, year=2023)
        for region in REGIONS:
            assert region in r.counts.columns


class TestAstcmb:
    def test_returns_astres(self, otis_df):
        a = astcmb(otis_df)
        assert isinstance(a, AstRes)

    def test_complexity_range(self, otis_df):
        a = astcmb(otis_df)
        assert a.data["ac"].min() >= 0
        assert a.data["ac"].max() <= 8

    def test_summary_has_counts(self, otis_df):
        a = astcmb(otis_df)
        assert "n_persons" in a.summary.columns


class TestVolat:
    def test_returns_volres(self, otis_df):
        v = volat(otis_df)
        assert isinstance(v, VolRes)
        assert v.mean > 0

    def test_data_has_vm(self, otis_df):
        v = volat(otis_df)
        assert "vm" in v.data.columns


class TestRctrnd:
    def test_returns_df(self, otis_df):
        t = rctrnd(otis_df)
        assert isinstance(t, pd.DataFrame)
        assert "year" in t.columns
        assert "region" in t.columns
        assert "n_individuals" in t.columns


class TestOtdesc:
    def test_returns_dict(self, otis_df):
        d = otdesc(otis_df)
        assert isinstance(d, dict)
        assert "n_total" in d
        assert "n_records" in d
        assert d["n_total"] > 0


class TestOtdml:
    def test_returns_otdmlr(self, otis_df):
        r = otdml(otis_df, outcome="Y", treatment="D",
                  covariates=["gender", "age_category"])
        assert isinstance(r, OtDmlR)

    def test_has_ate_and_att(self, otis_df):
        r = otdml(otis_df)
        assert hasattr(r, "ate")
        assert hasattr(r, "att")
        assert hasattr(r, "ate_pval")
        assert r.n > 0

    def test_se_positive(self, otis_df):
        r = otdml(otis_df)
        assert r.ate_se >= 0
        assert r.att_se >= 0
