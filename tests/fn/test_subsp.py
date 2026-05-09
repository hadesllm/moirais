"""Tests for moirais.fn.subsp -- Subsampling inference."""

import numpy as np
import pytest

from moirais.fn.subsp import subsp


@pytest.fixture()
def sample_data():
    rng = np.random.default_rng(77)
    return rng.standard_normal(80)


def test_returns_dict(sample_data):
    result = subsp(sample_data, np.mean)
    assert isinstance(result, dict)
    for k in ("estimate", "se", "ci_lower", "ci_upper", "b", "n_subsamples"):
        assert k in result


def test_estimate_matches(sample_data):
    result = subsp(sample_data, np.mean)
    assert result["estimate"] == pytest.approx(np.mean(sample_data))


def test_ci_finite(sample_data):
    result = subsp(sample_data, np.mean)
    assert np.isfinite(result["ci_lower"])
    assert np.isfinite(result["ci_upper"])


def test_se_positive(sample_data):
    result = subsp(sample_data, np.mean)
    assert result["se"] > 0


def test_custom_b(sample_data):
    result = subsp(sample_data, np.mean, b=10)
    assert result["b"] == 10


def test_invalid_b(sample_data):
    with pytest.raises(ValueError, match="b must be"):
        subsp(sample_data, np.mean, b=1)


def test_invalid_alpha(sample_data):
    with pytest.raises(ValueError, match="alpha"):
        subsp(sample_data, np.mean, alpha=1.5)


def test_cheatsheet():
    from moirais.fn.subsp import cheatsheet
    assert "subsampling" in cheatsheet().lower()
