"""Tests for mdlsc.py - MDL score."""
import numpy as np
from moirais.fn.mdlsc import mdl_score_fn, mdlsc


def test_mdlsc_returns_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = mdl_score_fn(x, max_order=10)
    assert result.name == "mdl_score"
    assert result.extra["best_order"] >= 1


def test_mdlsc_scores_length():
    x = np.random.default_rng(42).standard_normal(256)
    result = mdl_score_fn(x, max_order=12)
    assert len(result.extra["scores"]) == 12


def test_mdlsc_alias():
    x = np.random.default_rng(42).standard_normal(128)
    result = mdlsc(x, max_order=5)
    assert result.name == "mdl_score"
