"""Tests for moirais.fn.drctr — Doubly robust cross-fitted ATE estimator."""

import numpy as np
import pytest

from moirais.fn.drctr import drctr


@pytest.fixture()
def synth():
    rng = np.random.default_rng(42)
    n = 400
    x = rng.standard_normal((n, 2))
    ps = 1 / (1 + np.exp(-(0.5 * x[:, 0] - 0.3 * x[:, 1])))
    t = rng.binomial(1, ps)
    y = 1.0 + 2.0 * t + 0.4 * x[:, 0] + rng.standard_normal(n) * 0.5
    return y, t, x


def test_returns_dict(synth):
    result = drctr(*synth)
    assert isinstance(result, dict)
    for key in ("ate", "se", "ci_lower", "ci_upper", "fold_estimates", "n_folds", "n", "method"):
        assert key in result


def test_ate_finite(synth):
    result = drctr(*synth)
    assert np.isfinite(result["ate"])
    assert np.isfinite(result["se"])


def test_se_positive(synth):
    result = drctr(*synth)
    assert result["se"] > 0


def test_ci_brackets_ate(synth):
    result = drctr(*synth)
    assert result["ci_lower"] <= result["ate"] <= result["ci_upper"]


def test_method_label(synth):
    result = drctr(*synth)
    assert result["method"] == "DR-TMLE"


def test_fold_estimates_length(synth):
    result = drctr(*synth, n_folds=3)
    assert len(result["fold_estimates"]) == 3
    assert result["n_folds"] == 3


def test_ate_near_truth(synth):
    result = drctr(*synth)
    assert abs(result["ate"] - 2.0) < 1.5


def test_deterministic_with_seed(synth):
    r1 = drctr(*synth, seed=99)
    r2 = drctr(*synth, seed=99)
    assert r1["ate"] == r2["ate"]
