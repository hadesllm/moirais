"""Tests for bicsc.py - BIC score."""
import numpy as np
from morie.fn.bicsc import bic_score_fn, bicsc


def test_bicsc_returns_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = bic_score_fn(x, max_order=10)
    assert result.name == "bic_score"
    assert result.extra["best_order"] >= 1


def test_bicsc_scores_length():
    x = np.random.default_rng(42).standard_normal(256)
    result = bic_score_fn(x, max_order=15)
    assert len(result.extra["scores"]) == 15


def test_bicsc_alias():
    x = np.random.default_rng(42).standard_normal(128)
    result = bicsc(x, max_order=5)
    assert result.name == "bic_score"
