"""Tests for morie.fn.ps_fit — propensity score estimation via logistic regression."""

import numpy as np
import pandas as pd
import pytest

from morie.fn.ps_fit import compute_propensity_scores


@pytest.fixture()
def synth_data():
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    prob = 1 / (1 + np.exp(-(0.5 * x1 - 0.3 * x2)))
    t = rng.binomial(1, prob)
    return pd.DataFrame({"x1": x1, "x2": x2, "treatment": t})


def test_returns_series(synth_data):
    ps = compute_propensity_scores(synth_data, treatment="treatment", covariates=["x1", "x2"])
    assert isinstance(ps, pd.Series)


def test_values_in_unit_interval(synth_data):
    ps = compute_propensity_scores(synth_data, treatment="treatment", covariates=["x1", "x2"])
    assert (ps > 0).all() and (ps < 1).all()


def test_length_matches_input(synth_data):
    ps = compute_propensity_scores(synth_data, treatment="treatment", covariates=["x1", "x2"])
    assert len(ps) == len(synth_data)


def test_works_with_categorical_covariate():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "cat": rng.choice(["a", "b", "c"], n),
        "treatment": rng.binomial(1, 0.4, n),
    })
    ps = compute_propensity_scores(df, treatment="treatment", covariates=["x1", "cat"])
    assert len(ps) == n
    assert (ps > 0).all() and (ps < 1).all()


def test_index_preserved(synth_data):
    df = synth_data.copy()
    df.index = range(100, 100 + len(df))
    ps = compute_propensity_scores(df, treatment="treatment", covariates=["x1", "x2"])
    assert list(ps.index) == list(df.index)
