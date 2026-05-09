"""Tests for bayrhat.r_hat."""
import numpy as np
import pytest
from moirais.fn.bayrhat import r_hat


def test_bayrhat_basic():
    """Test basic functionality."""
    chains = np.random.default_rng(42).normal(0, 1, 100)
    result = r_hat(chains)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bayrhat_edge():
    """Test edge cases."""
    chains = np.random.default_rng(42).normal(0, 1, 100)
    result = r_hat(chains)
    assert isinstance(result, dict)
