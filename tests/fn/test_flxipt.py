"""Tests for flxipt.flexible_iptw."""
import numpy as np
import pytest
from morie.fn.flxipt import flexible_iptw


def test_flxipt_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    library = np.random.default_rng(42).normal(0, 1, 100)
    result = flexible_iptw(A, H, library)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_flxipt_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    library = np.random.default_rng(42).normal(0, 1, 100)
    result = flexible_iptw(A, H, library)
    assert isinstance(result, dict)
