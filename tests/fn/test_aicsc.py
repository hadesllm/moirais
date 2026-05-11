"""Tests for aicsc.py - AIC score."""
import numpy as np
from morie.fn.aicsc import aic_score_fn, aicsc


def test_aicsc_returns_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = aic_score_fn(x, max_order=10)
    assert result.name == "aic_score"
    assert result.extra["best_order"] >= 1
    assert result.extra["best_order"] <= 10


def test_aicsc_scores_length():
    x = np.random.default_rng(42).standard_normal(256)
    result = aic_score_fn(x, max_order=15)
    assert len(result.extra["scores"]) == 15


def test_aicsc_alias():
    x = np.random.default_rng(42).standard_normal(128)
    result = aicsc(x, max_order=5)
    assert result.name == "aic_score"
