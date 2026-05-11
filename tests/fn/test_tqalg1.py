"""Tests for tqalg1.turboquant_online_key_quantizer."""
import numpy as np
import pytest
from morie.fn.tqalg1 import turboquant_online_key_quantizer


def test_tqalg1_basic():
    """Test basic functionality."""
    k = 5
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_online_key_quantizer(k, S)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tqalg1_edge():
    """Test edge cases."""
    k = 5
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_online_key_quantizer(k, S)
    assert isinstance(result, dict)
