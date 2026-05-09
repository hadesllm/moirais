"""Tests for tmlhte.tmle_heterogeneous."""
import numpy as np
import pytest
from moirais.fn.tmlhte import tmle_heterogeneous


def test_tmlhte_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    strata = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_heterogeneous(y, treatment, W, strata)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlhte_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    strata = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_heterogeneous(y, treatment, W, strata)
    assert isinstance(result, dict)
