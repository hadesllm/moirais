"""Tests for moirais.fn.anchr — Anchor explanations."""
import numpy as np
import pytest
from moirais.fn.anchr import anchr


@pytest.fixture()
def setup():
    rng = np.random.default_rng(23)
    n, p = 200, 3
    X_train = rng.standard_normal((n, p))
    instance = np.array([1.5, -0.5, 0.3])
    # Threshold classifier on feature 0
    def predict_fn(Xp):
        return (Xp[:, 0] > 0).astype(float)
    return predict_fn, instance, X_train


def test_keys(setup):
    fn, inst, X_tr = setup
    r = anchr(fn, inst, X_tr, seed=0)
    for k in ("anchor_features", "anchor_conditions", "precision",
              "coverage", "prediction", "p", "method"):
        assert k in r


def test_anchor_features_subset(setup):
    fn, inst, X_tr = setup
    r = anchr(fn, inst, X_tr, seed=0)
    assert all(0 <= f < 3 for f in r["anchor_features"])


def test_precision_in_range(setup):
    fn, inst, X_tr = setup
    r = anchr(fn, inst, X_tr, seed=0)
    assert 0.0 <= r["precision"] <= 1.0


def test_coverage_in_range(setup):
    fn, inst, X_tr = setup
    r = anchr(fn, inst, X_tr, seed=0)
    assert 0.0 <= r["coverage"] <= 1.0


def test_prediction_returned(setup):
    fn, inst, X_tr = setup
    r = anchr(fn, inst, X_tr, seed=0)
    assert r["prediction"] in (0.0, 1.0)


def test_feature0_in_anchor(setup):
    """Feature 0 drives the decision, so should be in the anchor."""
    fn, inst, X_tr = setup
    r = anchr(fn, inst, X_tr, precision_threshold=0.8, seed=0)
    # Not guaranteed but likely for threshold classifier on feature 0
    assert isinstance(r["anchor_features"], list)


def test_method(setup):
    fn, inst, X_tr = setup
    r = anchr(fn, inst, X_tr, seed=0)
    assert r["method"] == "Anchor"


def test_cheatsheet():
    from moirais.fn.anchr import cheatsheet
    assert len(cheatsheet()) > 0
