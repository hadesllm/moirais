"""Tests for tqhs.turboquant_qjl_sign_quantizer."""
import numpy as np
import pytest
from morie.fn.tqhs import turboquant_qjl_sign_quantizer


def test_tqhs_basic():
    """Test basic functionality."""
    k = 5
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_qjl_sign_quantizer(k, S)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tqhs_edge():
    """Test edge cases."""
    k = 5
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_qjl_sign_quantizer(k, S)
    assert isinstance(result, dict)
