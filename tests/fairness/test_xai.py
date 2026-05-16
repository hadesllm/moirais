# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for morie.fairness.xai — the model-agnostic explainer suite.

Each explainer is checked against a model whose behaviour is known in
closed form: a model that uses only one feature, a linear model, an
additive model.
"""
import numpy as np
import pytest

from morie.fairness.xai import (
    xai_ale,
    xai_ceteris_paribus,
    xai_partial_dependence,
    xai_permutation_importance,
    xai_shap_values,
)

NAMES = ["a", "b", "c"]


def _X(seed=0, n=300):
    return np.random.default_rng(seed).normal(size=(n, 3))


# ── permutation importance ──────────────────────────────────────────

def test_permutation_importance_finds_the_used_feature():
    res = xai_permutation_importance(lambda A: A[:, 0], _X(),
                                     feature_names=NAMES, seed=1)
    imp = res.payload["importances"]
    assert res.payload["ranking"][0] == "a"
    assert imp["a"] > 0.1
    assert imp["b"] == pytest.approx(0.0, abs=1e-9)
    assert imp["c"] == pytest.approx(0.0, abs=1e-9)


def test_permutation_importance_flags_protected_feature():
    res = xai_permutation_importance(lambda A: A[:, 0], _X(),
                                     feature_names=["race", "b", "c"],
                                     protected=["race"], seed=1)
    assert res.payload["protected_ranks"]["race"] == 1
    assert any("protected" in w for w in res.warnings)


def test_permutation_importance_unused_protected_no_warning():
    # race is the unused feature 2 -> ranks last -> no bias warning
    res = xai_permutation_importance(lambda A: A[:, 0], _X(),
                                     feature_names=["a", "b", "race"],
                                     protected=["race"], seed=1)
    assert res.payload["protected_ranks"]["race"] == 3
    assert res.warnings == []


# ── partial dependence ──────────────────────────────────────────────

def test_partial_dependence_linear_model():
    res = xai_partial_dependence(lambda A: 2.0 * A[:, 0], _X(), "a",
                                 feature_names=NAMES, grid_size=12)
    grid, pd = res.payload["grid"], res.payload["pd"]
    assert np.allclose(pd, 2.0 * grid, atol=1e-6)


# ── ALE ─────────────────────────────────────────────────────────────

def test_ale_linear_model_range():
    X = _X()
    res = xai_ale(lambda A: 2.0 * A[:, 0], X, "a", feature_names=NAMES,
                  n_bins=8)
    feat_range = float(X[:, 0].max() - X[:, 0].min())
    assert res.payload["value"] == pytest.approx(2.0 * feat_range,
                                                 rel=0.05)


# ── ceteris paribus ─────────────────────────────────────────────────

def test_ceteris_paribus_linear_model():
    X = _X()
    res = xai_ceteris_paribus(lambda A: 3.0 * A[:, 1], X[0], "b",
                              X_ref=X, feature_names=NAMES, grid_size=12)
    grid, prof = res.payload["grid"], res.payload["profile"]
    assert np.allclose(prof, 3.0 * grid, atol=1e-6)


# ── SHAP ────────────────────────────────────────────────────────────

def test_shap_additive_model_sums_to_prediction_gap():
    X = _X(seed=2, n=200)
    res = xai_shap_values(lambda A: A.sum(axis=1), X[0], X,
                          feature_names=NAMES, n_samples=20, seed=3)
    total = sum(res.payload["shap_values"].values())
    gap = res.payload["prediction"] - res.payload["background_mean"]
    assert total == pytest.approx(gap, abs=1e-6)


def test_shap_additive_values_equal_centred_features():
    X = _X(seed=4, n=300)
    x = X[0]
    res = xai_shap_values(lambda A: A.sum(axis=1), x, X,
                          feature_names=NAMES, n_samples=20, seed=5)
    shap = res.payload["shap_values"]
    for j, nm in enumerate(NAMES):
        assert shap[nm] == pytest.approx(x[j] - X[:, j].mean(), abs=1e-6)


# ── input validation ────────────────────────────────────────────────

def test_xai_bad_X_shape_raises():
    with pytest.raises(ValueError):
        xai_partial_dependence(lambda A: A[:, 0], np.zeros(5), 0)


def test_xai_unknown_feature_name_raises():
    with pytest.raises(ValueError):
        xai_partial_dependence(lambda A: A[:, 0], _X(), "nope",
                               feature_names=NAMES)
