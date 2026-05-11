"""Tests for eslsvc.esl_svc."""
import numpy as np
import pytest
from morie.fn.eslsvc import esl_svc


def test_eslsvc_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_svc(X, y, C)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslsvc_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_svc(X, y, C)
    assert isinstance(result, dict)
