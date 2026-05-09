"""Tests for catsc.py - CAT score."""
import numpy as np
from moirais.fn.catsc import cat_score_fn, catsc


def test_catsc_returns_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = cat_score_fn(x, max_order=10)
    assert result.name == "cat_score"
    assert result.extra["best_order"] >= 1


def test_catsc_scores_length():
    x = np.random.default_rng(42).standard_normal(256)
    result = cat_score_fn(x, max_order=8)
    assert len(result.extra["scores"]) == 8


def test_catsc_alias():
    x = np.random.default_rng(42).standard_normal(128)
    result = catsc(x, max_order=5)
    assert result.name == "cat_score"
