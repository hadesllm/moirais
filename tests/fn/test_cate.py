"""Tests for morie.fn.cate — Conditional Average Treatment Effect via meta-learners."""

import numpy as np
import pandas as pd
import pytest

from morie.fn.cate import estimate_cate


@pytest.fixture()
def synth_data():
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    prob = 1 / (1 + np.exp(-(0.5 * x1)))
    t = rng.binomial(1, prob)
    y = 0.4 * t * x1 + 0.3 * x2 + rng.standard_normal(n) * 0.3
    return pd.DataFrame({"x1": x1, "x2": x2, "treatment": t, "outcome": y})


def test_returns_series(synth_data):
    result = estimate_cate(
        synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"]
    )
    assert isinstance(result, pd.Series)


def test_cate_values_are_finite(synth_data):
    result = estimate_cate(
        synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"]
    )
    assert np.all(np.isfinite(result.values))


def test_length_matches_input(synth_data):
    result = estimate_cate(
        synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"]
    )
    assert len(result) == len(synth_data)


def test_s_learner(synth_data):
    result = estimate_cate(
        synth_data,
        treatment="treatment",
        outcome="outcome",
        covariates=["x1", "x2"],
        meta_learner="s_learner",
    )
    assert isinstance(result, pd.Series)
    assert np.all(np.isfinite(result.values))


def test_invalid_meta_learner_raises(synth_data):
    with pytest.raises(ValueError, match="Unknown meta_learner"):
        estimate_cate(
            synth_data,
            treatment="treatment",
            outcome="outcome",
            covariates=["x1", "x2"],
            meta_learner="x_learner",
        )
