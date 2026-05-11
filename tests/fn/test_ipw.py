"""Tests for morie.fn.ipw — Inverse Probability of Treatment Weighting."""

import numpy as np
import pandas as pd
import pytest

from morie.fn.ipw import calculate_ipw_weights


@pytest.fixture()
def synth_data():
    rng = np.random.default_rng(42)
    n = 200
    ps = rng.uniform(0.1, 0.9, n)
    t = rng.binomial(1, ps)
    return pd.DataFrame({"treatment": t, "ps": ps})


def test_returns_series(synth_data):
    w = calculate_ipw_weights(synth_data, treatment="treatment", ps_col="ps")
    assert isinstance(w, pd.Series)


def test_weights_positive(synth_data):
    w = calculate_ipw_weights(synth_data, treatment="treatment", ps_col="ps")
    assert (w > 0).all()


def test_treated_weight_formula(synth_data):
    """For treated units, unstabilised weight = 1/e(X). For controls, 1/(1-e(X))."""
    w = calculate_ipw_weights(synth_data, treatment="treatment", ps_col="ps")
    ps_clipped = synth_data["ps"].clip(lower=0.01, upper=0.99)
    t = synth_data["treatment"]
    expected = (t / ps_clipped) + ((1 - t) / (1 - ps_clipped))
    np.testing.assert_allclose(w.values, expected.values, atol=1e-10)


def test_stabilized_weights(synth_data):
    w = calculate_ipw_weights(synth_data, treatment="treatment", ps_col="ps", stabilized=True)
    assert isinstance(w, pd.Series)
    assert (w > 0).all()


def test_trim_quantiles(synth_data):
    w_raw = calculate_ipw_weights(synth_data, treatment="treatment", ps_col="ps")
    w_trim = calculate_ipw_weights(
        synth_data, treatment="treatment", ps_col="ps", trim_quantiles=(0.05, 0.95)
    )
    assert w_trim.max() <= w_raw.max()
    assert w_trim.min() >= w_raw.min()
