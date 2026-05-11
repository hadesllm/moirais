"""Tests for morie.psymet — psychometric analysis functions."""

import os

import numpy as np
import pandas as pd
import pytest

from morie.psymet import (
    RlbRes, OmgRes, KmoRes, BrtRes,
    crba, mcdo, kmo, bart, paran, splhf, itcor, adel, idisc, crel, ave,
)


@pytest.fixture
def mapq():
    """Load MAPQII from morie.db (works in CI) or xlsx fallback."""
    try:
        from morie.data import load_dataset
        df = load_dataset("mapq")
        # load_dataset returns the full survey — filter to MAPQII 20 items
        mapq_cols = [f"{s}{i}" for s in ("EE","EA","UA","ER") for i in range(1,6)]
        if all(c in df.columns for c in mapq_cols):
            return df[mapq_cols]
        return df.select_dtypes(include="number")
    except Exception:
        pass
    # Fallback to xlsx if available
    path = "data/datasets/vsr/TKARONTOMAPQ.xlsx"
    if os.path.exists(path):
        return pd.read_excel(path, sheet_name="MAPQII")
    pytest.skip("MAPQ dataset not available")


@pytest.fixture
def likert():
    """Synthetic 5-point Likert data with known structure."""
    rng = np.random.default_rng(42)
    n, k = 100, 10
    factor = rng.standard_normal((n, 1))
    items = factor @ rng.standard_normal((1, k)) + rng.standard_normal((n, k)) * 0.5
    items = np.clip(np.round(items * 0.8 + 3), 1, 5)
    return pd.DataFrame(items, columns=[f"q{i}" for i in range(k)])


class TestCrba:
    def test_returns_rlbres(self, likert):
        r = crba(likert)
        assert isinstance(r, RlbRes)

    def test_mapq_alpha_high(self, mapq):
        r = crba(mapq)
        assert r.raw > 0.90
        assert r.std > 0.90

    def test_ci_contains_point(self, likert):
        r = crba(likert)
        assert r.ci_lo <= r.raw <= r.ci_hi

    def test_constant_data_nan(self):
        r = crba(np.ones((20, 5)))
        assert np.isnan(r.raw)

    def test_single_item_nan(self):
        r = crba(np.random.randn(20, 1))
        assert np.isnan(r.raw)


class TestMcdo:
    def test_returns_omgres(self, likert):
        o = mcdo(likert)
        assert isinstance(o, OmgRes)

    def test_mapq_omega_high(self, mapq):
        o = mcdo(mapq, nf=4)
        assert o.total > 0.90

    def test_omega_geq_alpha(self, likert):
        o = mcdo(likert)
        assert o.total >= o.alpha - 0.01  # omega >= alpha (approx)


class TestKmo:
    def test_returns_kmores(self, likert):
        k = kmo(likert)
        assert isinstance(k, KmoRes)

    def test_mapq_kmo_adequate(self, mapq):
        k = kmo(mapq)
        assert k.msa > 0.6  # adequate for FA

    def test_per_item_msa(self, mapq):
        k = kmo(mapq)
        assert len(k.items) == mapq.shape[1]
        for v in k.items.values():
            assert 0 <= v <= 1


class TestBart:
    def test_returns_brtres(self, likert):
        b = bart(likert)
        assert isinstance(b, BrtRes)

    def test_mapq_significant(self, mapq):
        b = bart(mapq)
        assert b.pval < 0.001
        assert b.df == mapq.shape[1] * (mapq.shape[1] - 1) // 2


class TestParan:
    def test_returns_int(self, likert):
        nf = paran(likert)
        assert isinstance(nf, int)
        assert nf >= 1

    def test_structured_data_few_factors(self, likert):
        nf = paran(likert)
        assert nf <= 5


class TestSplhf:
    def test_returns_float(self, likert):
        sh = splhf(likert)
        assert isinstance(sh, float)

    def test_methods(self, likert):
        sh1 = splhf(likert, method="first_last")
        sh2 = splhf(likert, method="odd_even")
        assert -1 <= sh1 <= 1
        assert -1 <= sh2 <= 1


class TestItcor:
    def test_returns_df(self, likert):
        r = itcor(likert)
        assert isinstance(r, pd.DataFrame)
        assert "item" in r.columns
        assert "r_corr" in r.columns
        assert len(r) == likert.shape[1]

    def test_mapq_all_positive(self, mapq):
        r = itcor(mapq)
        assert (r["r_corr"] > 0).all()


class TestAdel:
    def test_returns_df(self, likert):
        r = adel(likert)
        assert isinstance(r, pd.DataFrame)
        assert len(r) == likert.shape[1]


class TestIdisc:
    def test_returns_df(self, likert):
        r = idisc(likert)
        assert isinstance(r, pd.DataFrame)
        assert "d" in r.columns
        assert len(r) == likert.shape[1]


class TestCrel:
    def test_high_loadings(self):
        assert crel(np.array([0.8, 0.9, 0.7, 0.85])) > 0.8

    def test_low_loadings(self):
        assert crel(np.array([0.2, 0.1, 0.15])) < 0.5


class TestAve:
    def test_adequate(self):
        assert ave(np.array([0.8, 0.9, 0.7])) >= 0.5

    def test_inadequate(self):
        assert ave(np.array([0.3, 0.2, 0.25])) < 0.5
