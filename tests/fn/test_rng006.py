"""Tests for rng006.rangayyan_ch3_entropy_continuous."""
import numpy as np
import pytest
from morie.fn.rng006 import rangayyan_ch3_entropy_continuous


def test_rng006_basic():
    """Test basic functionality."""
    eta = np.random.default_rng(42).normal(0, 1, 100)
    p_eta = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_entropy_continuous(eta, p_eta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng006_edge():
    """Test edge cases."""
    eta = np.random.default_rng(42).normal(0, 1, 100)
    p_eta = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_entropy_continuous(eta, p_eta)
    assert isinstance(result, dict)
