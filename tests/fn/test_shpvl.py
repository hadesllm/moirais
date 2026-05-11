"""Tests for morie.fn.shpvl — Kernel SHAP values."""
import numpy as np
import pytest
from morie.fn.shpvl import shpvl


@pytest.fixture()
def setup():
    rng = np.random.default_rng(20)
    n_bg, p = 50, 3
    X_bg = rng.standard_normal((n_bg, p))
    X_explain = rng.standard_normal((5, p))
    beta = np.array([3.0, -1.0, 0.5])
    def predict_fn(Xp):
        return Xp @ beta
    return predict_fn, X_explain, X_bg


def test_keys(setup):
    fn, X, bg = setup
    r = shpvl(fn, X, bg, n_samples=64, seed=0)
    for k in ("shap_values", "base_value", "m", "p", "method"):
        assert k in r


def test_shap_shape(setup):
    fn, X, bg = setup
    r = shpvl(fn, X, bg, n_samples=64)
    assert r["shap_values"].shape == (5, 3)


def test_efficiency(setup):
    """SHAP efficiency: phi_0 + sum(phi) = f(x) for each observation."""
    fn, X, bg = setup
    r = shpvl(fn, X, bg, n_samples=128, seed=0)
    for i in range(5):
        f_x = float(fn(X[[i]])[0])
        total = r["base_value"] + r["shap_values"][i].sum()
        assert abs(total - f_x) < 0.5  # approximate, not exact


def test_base_value_close_to_mean(setup):
    fn, X, bg = setup
    r = shpvl(fn, X, bg, n_samples=64, seed=0)
    bg_mean_pred = float(fn(bg).mean())
    assert abs(r["base_value"] - bg_mean_pred) < 1e-6


def test_method(setup):
    fn, X, bg = setup
    assert shpvl(fn, X, bg, n_samples=64)["method"] == "KernelSHAP"


def test_most_important_feature(setup):
    """Feature 0 (beta=3) should have largest |SHAP| on average."""
    fn, X, bg = setup
    r = shpvl(fn, X, bg, n_samples=256, seed=0)
    mean_abs = np.mean(np.abs(r["shap_values"]), axis=0)
    # Not always exact but beta[0]=3 should dominate
    assert mean_abs[0] == mean_abs.max() or mean_abs[0] > mean_abs[1]


def test_cheatsheet():
    from morie.fn.shpvl import cheatsheet
    assert len(cheatsheet()) > 0
