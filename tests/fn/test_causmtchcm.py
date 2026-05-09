"""Tests for causmtchcm.causal_caliper_matching."""
import numpy as np
import pytest
from moirais.fn.causmtchcm import causal_caliper_matching


def test_causmtchcm_basic():
    """Test basic functionality."""
    ps = np.random.default_rng(42).normal(0, 1, 100)
    treat = np.random.default_rng(42).normal(0, 1, 100)
    caliper = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_caliper_matching(ps, treat, caliper)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causmtchcm_edge():
    """Test edge cases."""
    ps = np.random.default_rng(42).normal(0, 1, 100)
    treat = np.random.default_rng(42).normal(0, 1, 100)
    caliper = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_caliper_matching(ps, treat, caliper)
    assert isinstance(result, dict)
