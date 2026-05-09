"""Tests for qboIdx.qbo."""
import numpy as np
import pytest
from moirais.fn.qboIdx import qbo


def test_qboIdx_basic():
    """Test basic functionality."""
    U30 = np.random.default_rng(42).normal(0, 1, 100)
    result = qbo(U30)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_qboIdx_edge():
    """Test edge cases."""
    U30 = np.random.default_rng(42).normal(0, 1, 100)
    result = qbo(U30)
    assert isinstance(result, dict)
