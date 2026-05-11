"""Tests for morie.fn.effsc — Efficient score function."""

import numpy as np
import pytest

from morie.fn.effsc import effsc


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
    result = effsc(*synth)
    assert isinstance(result, dict)
    for key in ("scores", "estimate", "se", "ci_lower", "ci_upper", "info_bound", "n"):
        assert key in result


def test_scores_shape(synth):
    result = effsc(*synth)
    assert result["scores"].shape == (len(synth[0]),)


def test_estimate_finite(synth):
    result = effsc(*synth)
    assert np.isfinite(result["estimate"])
    assert np.isfinite(result["se"])


def test_se_positive(synth):
    result = effsc(*synth)
    assert result["se"] > 0


def test_info_bound_positive(synth):
    result = effsc(*synth)
    assert result["info_bound"] > 0


def test_ci_brackets(synth):
    result = effsc(*synth)
    assert result["ci_lower"] <= result["estimate"] <= result["ci_upper"]


def test_method_label(synth):
    result = effsc(*synth)
    assert result["method"] == "EfficientScore"
