"""Tests for moirais.fn.icerc — ICE curves."""
import numpy as np
import pytest
from moirais.fn.icerc import icerc


@pytest.fixture()
def setup():
    rng = np.random.default_rng(18)
    n, p = 80, 3
    X = rng.standard_normal((n, p))
    # Linear model: beta = [1, 2, -1]
    beta = np.array([1.0, 2.0, -1.0])
    def predict_fn(Xp):
        return Xp @ beta
    return predict_fn, X


def test_keys(setup):
    fn, X = setup
    r = icerc(fn, X, feature_idx=0, grid_points=10)
    for k in ("ice", "pdp", "grid", "feature_idx", "n", "G", "method"):
        assert k in r


def test_ice_shape(setup):
    fn, X = setup
    r = icerc(fn, X, feature_idx=1, grid_points=15)
    assert r["ice"].shape == (80, 15)


def test_pdp_shape(setup):
    fn, X = setup
    r = icerc(fn, X, feature_idx=0, grid_points=20)
    assert r["pdp"].shape == (20,)


def test_center_shifts_to_zero(setup):
    fn, X = setup
    r = icerc(fn, X, feature_idx=0, grid_points=10, center=True)
    np.testing.assert_allclose(r["ice"][:, 0], 0.0, atol=1e-10)


def test_not_centered(setup):
    fn, X = setup
    r = icerc(fn, X, feature_idx=0, grid_points=10, center=False)
    assert r["method"] == "ICE"


def test_linear_pdp_is_monotone(setup):
    fn, X = setup
    r = icerc(fn, X, feature_idx=0, grid_points=20, center=False)
    # beta[0]=1 > 0, so PDP should be increasing
    assert r["pdp"][-1] > r["pdp"][0]


def test_custom_grid(setup):
    fn, X = setup
    grid = np.array([0.0, 0.5, 1.0])
    r = icerc(fn, X, feature_idx=0, grid_values=grid)
    assert r["G"] == 3


def test_cheatsheet():
    from moirais.fn.icerc import cheatsheet
    assert len(cheatsheet()) > 0
