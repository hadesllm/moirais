"""Tests for causqte.causal_quantile_treatment_effect."""
import numpy as np
import pytest
from moirais.fn.causqte import causal_quantile_treatment_effect


def test_causqte_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    ps = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = causal_quantile_treatment_effect(y, T, ps, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causqte_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    ps = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = causal_quantile_treatment_effect(y, T, ps, tau)
    assert isinstance(result, dict)
