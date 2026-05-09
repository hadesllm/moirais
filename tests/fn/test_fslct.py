"""Tests for feature_select_variance."""
import numpy as np
from moirais.fn.fslct import feature_select_variance, fslct


def test_basic():
    X = np.array([[1, 0, 100], [2, 0, 200], [3, 0, 300]])
    r = feature_select_variance(X, threshold=0.0)
    assert 1 not in r.extra["selected_indices"]
    assert 0 in r.extra["selected_indices"]
    assert 2 in r.extra["selected_indices"]
    assert r.extra["n_selected"] == 2


def test_alias():
    assert fslct is feature_select_variance


def test_all_constant():
    X = np.ones((5, 3))
    r = feature_select_variance(X, threshold=0.0)
    assert r.extra["n_selected"] == 0
