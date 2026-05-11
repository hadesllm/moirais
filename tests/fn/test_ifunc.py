"""Tests for morie.fn.ifunc — Influence function computation."""

import numpy as np
import pytest

from morie.fn.ifunc import ifunc


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
    result = ifunc(*synth)
    assert isinstance(result, dict)
    for key in ("influence_values", "estimate", "se", "ci_lower", "ci_upper", "n", "functional"):
        assert key in result


def test_influence_values_shape(synth):
    result = ifunc(*synth)
    assert result["influence_values"].shape == (len(synth[0]),)


def test_ate_finite(synth):
    result = ifunc(*synth, functional="ate")
    assert np.isfinite(result["estimate"])
    assert np.isfinite(result["se"])


def test_se_positive(synth):
    result = ifunc(*synth)
    assert result["se"] > 0


def test_ci_brackets(synth):
    result = ifunc(*synth)
    assert result["ci_lower"] <= result["estimate"] <= result["ci_upper"]


def test_att_functional(synth):
    result = ifunc(*synth, functional="att")
    assert result["functional"] == "att"
    assert np.isfinite(result["estimate"])


def test_mean_functional(synth):
    result = ifunc(*synth, functional="mean")
    assert result["functional"] == "mean"
    assert np.isfinite(result["estimate"])


def test_unknown_functional_raises(synth):
    with pytest.raises(ValueError, match="Unknown functional"):
        ifunc(*synth, functional="bad")


def test_influence_mean_near_zero(synth):
    result = ifunc(*synth)
    assert abs(np.mean(result["influence_values"])) < 1.0
