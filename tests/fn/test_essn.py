"""Tests for morie.fn.essn -- Effective sample size."""

import numpy as np
from morie.fn.essn import effective_sample_size


def test_returns_dict():
    samples = np.random.default_rng(42).standard_normal(1000)
    result = effective_sample_size(samples)
    assert isinstance(result, dict)
    assert "ess" in result


def test_iid_ess_near_n():
    samples = np.random.default_rng(42).standard_normal(1000)
    result = effective_sample_size(samples)
    assert result["ess"] > 500


def test_ess_positive():
    samples = np.random.default_rng(42).standard_normal(100)
    result = effective_sample_size(samples)
    assert result["ess"] > 0
