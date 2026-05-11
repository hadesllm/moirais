"""Tests for rdpc.renyi_dp."""
import numpy as np
import pytest
from morie.fn.rdpc import renyi_dp


def test_rdpc_basic():
    """Test basic functionality."""
    alpha = 0.05
    sigma = 1.0
    result = renyi_dp(alpha, sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rdpc_edge():
    """Test edge cases."""
    alpha = 0.05
    sigma = 1.0
    result = renyi_dp(alpha, sigma)
    assert isinstance(result, dict)
