"""Tests for moirais.fn.limex — LIME explanations."""
import numpy as np
import pytest
from moirais.fn.limex import limex


@pytest.fixture()
def setup():
    rng = np.random.default_rng(21)
    n, p = 200, 4
    X_train = rng.standard_normal((n, p))
    instance = rng.standard_normal(p)
    beta = np.array([2.0, -1.5, 0.5, 0.1])
    def predict_fn(Xp):
        return Xp @ beta
    return predict_fn, instance, X_train, beta


def test_keys(setup):
    fn, inst, X_tr, _ = setup
    r = limex(fn, inst, X_tr, n_samples=200, seed=0)
    for k in ("coefficients", "intercept", "local_r2", "weights", "p", "method"):
        assert k in r


def test_coeff_shape(setup):
    fn, inst, X_tr, _ = setup
    r = limex(fn, inst, X_tr, n_samples=200)
    assert r["coefficients"].shape == (4,)


def test_local_r2_in_range(setup):
    fn, inst, X_tr, _ = setup
    r = limex(fn, inst, X_tr, n_samples=500, seed=0)
    assert -0.1 <= r["local_r2"] <= 1.1


def test_most_important_feature(setup):
    """Feature 0 (beta=2) should dominate."""
    fn, inst, X_tr, beta = setup
    r = limex(fn, inst, X_tr, n_samples=500, seed=0)
    abs_coef = np.abs(r["coefficients"])
    assert abs_coef[0] == abs_coef.max()


def test_n_features_sparsity(setup):
    fn, inst, X_tr, _ = setup
    r = limex(fn, inst, X_tr, n_samples=300, n_features=2, seed=0)
    nonzero = np.sum(r["coefficients"] != 0)
    assert nonzero <= 2


def test_method(setup):
    fn, inst, X_tr, _ = setup
    assert limex(fn, inst, X_tr, n_samples=100)["method"] == "LIME"


def test_weights_positive(setup):
    fn, inst, X_tr, _ = setup
    r = limex(fn, inst, X_tr, n_samples=100)
    assert np.all(r["weights"] > 0)


def test_cheatsheet():
    from moirais.fn.limex import cheatsheet
    assert len(cheatsheet()) > 0
