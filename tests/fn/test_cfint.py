"""Tests for moirais.fn.cfint — Counterfactual explanation."""
import numpy as np
import pytest
from moirais.fn.cfint import cfint


@pytest.fixture()
def setup():
    rng = np.random.default_rng(22)
    n, p = 100, 3
    X_train = rng.standard_normal((n, p))
    instance = rng.standard_normal(p)
    # Model: predict 1 if sum > 0
    def predict_fn(Xp):
        return (Xp.sum(axis=1) > 0).astype(float)
    return predict_fn, instance, X_train


def test_keys(setup):
    fn, inst, X_tr = setup
    r = cfint(fn, inst, X_tr, n_cf=3, max_iter=30, seed=0)
    for k in ("counterfactuals", "distances", "predictions",
              "instance", "target_class", "p", "method"):
        assert k in r


def test_cf_shape(setup):
    fn, inst, X_tr = setup
    r = cfint(fn, inst, X_tr, n_cf=3, max_iter=20, seed=0)
    assert r["counterfactuals"].shape[1] == 3


def test_distances_nonneg(setup):
    fn, inst, X_tr = setup
    r = cfint(fn, inst, X_tr, n_cf=2, max_iter=20, seed=0)
    assert np.all(r["distances"] >= 0)


def test_distances_sorted(setup):
    fn, inst, X_tr = setup
    r = cfint(fn, inst, X_tr, n_cf=3, max_iter=20, seed=0)
    dists = r["distances"]
    assert np.all(dists[:-1] <= dists[1:] + 1e-9)


def test_predictions_finite(setup):
    fn, inst, X_tr = setup
    r = cfint(fn, inst, X_tr, n_cf=2, max_iter=20, seed=0)
    assert np.all(np.isfinite(r["predictions"]))


def test_method(setup):
    fn, inst, X_tr = setup
    r = cfint(fn, inst, X_tr, n_cf=1, max_iter=10, seed=0)
    assert "counterfactual" in r["method"]


def test_p_correct(setup):
    fn, inst, X_tr = setup
    r = cfint(fn, inst, X_tr, n_cf=1, max_iter=5, seed=0)
    assert r["p"] == 3


def test_cheatsheet():
    from moirais.fn.cfint import cheatsheet
    assert len(cheatsheet()) > 0
