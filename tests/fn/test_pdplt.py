"""Tests for morie.fn.pdplt — Partial dependence plot values."""
import numpy as np
import pytest
from morie.fn.pdplt import pdplt


@pytest.fixture()
def setup():
    rng = np.random.default_rng(19)
    n, p = 100, 3
    X = rng.standard_normal((n, p))
    beta = np.array([2.0, -1.0, 0.5])
    def predict_fn(Xp):
        return Xp @ beta
    return predict_fn, X


def test_keys_1d(setup):
    fn, X = setup
    r = pdplt(fn, X, feature_idx=0)
    for k in ("pdp", "grid", "feature_idx", "n", "method"):
        assert k in r


def test_pdp_shape_1d(setup):
    fn, X = setup
    r = pdplt(fn, X, feature_idx=0, grid_points=20)
    assert r["pdp"].shape == (20,)


def test_grid_shape_1d(setup):
    fn, X = setup
    r = pdplt(fn, X, feature_idx=0, grid_points=25)
    assert r["grid"].shape == (25,)


def test_pdp_monotone_positive_beta(setup):
    fn, X = setup
    r = pdplt(fn, X, feature_idx=0, grid_points=15)
    # beta[0]=2 > 0: PDP should increase
    assert r["pdp"][-1] > r["pdp"][0]


def test_pdp_monotone_negative_beta(setup):
    fn, X = setup
    r = pdplt(fn, X, feature_idx=1, grid_points=15)
    # beta[1]=-1 < 0: PDP should decrease
    assert r["pdp"][-1] < r["pdp"][0]


def test_2d_pdp(setup):
    fn, X = setup
    r = pdplt(fn, X, feature_idx=(0, 1), grid_points=8)
    assert r["pdp"].shape == (8, 8)
    assert len(r["grid"]) == 2


def test_method_label(setup):
    fn, X = setup
    assert pdplt(fn, X, feature_idx=0)["method"] == "PDP"


def test_cheatsheet():
    from morie.fn.pdplt import cheatsheet
    assert len(cheatsheet()) > 0
