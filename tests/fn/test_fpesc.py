"""Tests for fpesc.py - FPE score."""
import numpy as np
from morie.fn.fpesc import fpe_score_fn, fpesc


def test_fpesc_returns_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = fpe_score_fn(x, max_order=10)
    assert result.name == "fpe_score"
    assert result.extra["best_order"] >= 1


def test_fpesc_scores_positive():
    x = np.random.default_rng(42).standard_normal(256)
    result = fpe_score_fn(x, max_order=10)
    assert np.all(result.extra["scores"] > 0)


def test_fpesc_alias():
    x = np.random.default_rng(42).standard_normal(128)
    result = fpesc(x, max_order=5)
    assert result.name == "fpe_score"
