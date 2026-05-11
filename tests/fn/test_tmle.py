"""Tests for morie.fn.tmle — Targeted Minimum Loss-Based Estimation."""

import numpy as np
import pytest

from morie.fn.tmle import tmle


@pytest.fixture()
def synth():
    rng = np.random.default_rng(42)
    n = 300
    x = rng.standard_normal((n, 2))
    ps = 1 / (1 + np.exp(-(0.5 * x[:, 0] - 0.3 * x[:, 1])))
    t = rng.binomial(1, ps)
    y = (0.5 + 1.5 * t + 0.4 * x[:, 0] + rng.standard_normal(n) * 0.5 > 0.5).astype(float)
    return y, t, x


def test_returns_dict(synth):
    result = tmle(*synth)
    assert isinstance(result, dict)
    for key in ("ate", "se", "ci_lower", "ci_upper", "epsilon", "n", "method"):
        assert key in result


def test_ate_finite(synth):
    result = tmle(*synth)
    assert np.isfinite(result["ate"])
    assert np.isfinite(result["se"])


def test_se_positive(synth):
    result = tmle(*synth)
    assert result["se"] > 0


def test_ci_brackets_ate(synth):
    result = tmle(*synth)
    assert result["ci_lower"] <= result["ate"] <= result["ci_upper"]


def test_method_label(synth):
    result = tmle(*synth)
    assert result["method"] == "TMLE"


def test_linear_outcome(synth):
    y, t, x = synth
    y_cont = 1.0 + 2.0 * t + 0.3 * x[:, 0] + np.random.default_rng(7).standard_normal(len(t)) * 0.5
    result = tmle(y_cont, t, x, outcome_model="linear")
    assert np.isfinite(result["ate"])


def test_n_correct(synth):
    result = tmle(*synth)
    assert result["n"] == len(synth[0])
