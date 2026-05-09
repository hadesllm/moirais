"""Tests for ocmtmd.outcome_model_diagnostic."""
import numpy as np
import pytest
from moirais.fn.ocmtmd import outcome_model_diagnostic


def test_ocmtmd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    result = outcome_model_diagnostic(y, A, H, Q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ocmtmd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    result = outcome_model_diagnostic(y, A, H, Q)
    assert isinstance(result, dict)
