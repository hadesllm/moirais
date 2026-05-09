"""Tests for moirais.fn.ebmix -- Empirical Bayes mixture."""

import numpy as np
from moirais.fn.ebmix import eb_mixture


def test_returns_dict():
    z = np.random.default_rng(42).standard_normal(200)
    result = eb_mixture(z)
    assert isinstance(result, dict)
    assert "pi0" in result


def test_pi0_in_range():
    z = np.random.default_rng(42).standard_normal(200)
    result = eb_mixture(z)
    assert 0 < result["pi0"] <= 1


def test_local_fdr_length():
    z = np.random.default_rng(42).standard_normal(100)
    result = eb_mixture(z)
    assert len(result["local_fdr"]) == 100
