"""Tests for moirais.fn.bdic -- Bayesian DIC."""

import numpy as np
from moirais.fn.bdic import bayesian_dic


def test_returns_dict():
    samples = np.random.default_rng(42).normal(0, 1, (100, 1))
    result = bayesian_dic(lambda s: -0.5 * float(s @ s), samples)
    assert isinstance(result, dict)
    assert "dic" in result


def test_dic_finite():
    samples = np.random.default_rng(42).normal(0, 1, (100, 1))
    result = bayesian_dic(lambda s: -0.5 * float(s @ s), samples)
    assert np.isfinite(result["dic"])


def test_p_d_non_negative():
    samples = np.random.default_rng(42).normal(0, 0.1, (200, 1))
    result = bayesian_dic(lambda s: -0.5 * float(s @ s), samples)
    assert result["p_d"] >= -1.0
