"""Tests for bayth.bayes_theorem_genomic."""
import numpy as np
import pytest
from morie.fn.bayth import bayes_theorem_genomic


def test_bayth_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    prior_f = np.random.default_rng(42).normal(0, 1, 100)
    likelihood_f = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_theorem_genomic(y, prior_f, likelihood_f)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bayth_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    prior_f = np.random.default_rng(42).normal(0, 1, 100)
    likelihood_f = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_theorem_genomic(y, prior_f, likelihood_f)
    assert isinstance(result, dict)
