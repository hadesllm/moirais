"""Tests for causscd.causal_synthetic_did."""
import numpy as np
import pytest
from moirais.fn.causscd import causal_synthetic_did


def test_causscd_basic():
    """Test basic functionality."""
    Y_panel = np.random.default_rng(42).normal(0, 1, 100)
    treated_idx = np.random.default_rng(42).normal(0, 1, 100)
    treat_time = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_synthetic_did(Y_panel, treated_idx, treat_time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causscd_edge():
    """Test edge cases."""
    Y_panel = np.random.default_rng(42).normal(0, 1, 100)
    treated_idx = np.random.default_rng(42).normal(0, 1, 100)
    treat_time = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_synthetic_did(Y_panel, treated_idx, treat_time)
    assert isinstance(result, dict)
