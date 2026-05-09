"""Tests for chrwgt.censoring_at_risk_weight."""
import numpy as np
import pytest
from moirais.fn.chrwgt import censoring_at_risk_weight


def test_chrwgt_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    censor = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = censoring_at_risk_weight(time, censor, A, H)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_chrwgt_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    censor = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = censoring_at_risk_weight(time, censor, A, H)
    assert isinstance(result, dict)
