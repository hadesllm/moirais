"""Tests for morie.fn.onest — One-step semiparametric ATE estimator."""

import numpy as np
import pytest

from morie.fn.onest import onest


@pytest.fixture()
def synth():
    rng = np.random.default_rng(42)
    n = 300
    x = rng.standard_normal((n, 2))
    ps = 1 / (1 + np.exp(-(0.5 * x[:, 0] - 0.3 * x[:, 1])))
    t = rng.binomial(1, ps)
    y = 1.0 + 2.0 * t + 0.4 * x[:, 0] + rng.standard_normal(n) * 0.5
    return y, t, x


def test_returns_dict(synth):
    result = onest(*synth)
    assert isinstance(result, dict)
    for key in ("ate", "se", "ci_lower", "ci_upper", "initial_ate", "correction", "n", "method"):
        assert key in result


def test_ate_finite(synth):
    result = onest(*synth)
    assert np.isfinite(result["ate"])
    assert np.isfinite(result["se"])


def test_se_positive(synth):
    result = onest(*synth)
    assert result["se"] > 0


def test_ci_brackets_ate(synth):
    result = onest(*synth)
    assert result["ci_lower"] <= result["ate"] <= result["ci_upper"]


def test_method_label(synth):
    result = onest(*synth)
    assert result["method"] == "OneStep"


def test_with_initial_estimate(synth):
    result = onest(*synth, initial_estimate=1.5)
    assert result["initial_ate"] == 1.5
    assert np.isfinite(result["correction"])


def test_correction_is_small_for_good_init(synth):
    r1 = onest(*synth)
    r2 = onest(*synth, initial_estimate=r1["ate"])
    assert abs(r2["correction"]) < abs(r1["correction"]) + 0.5
