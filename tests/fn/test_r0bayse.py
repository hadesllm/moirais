"""Tests for r0bayse.basic_reproduction."""
import numpy as np
import pytest
from morie.fn.r0bayse import basic_reproduction


def test_r0bayse_basic():
    """Test basic functionality."""
    beta = 0.8
    gamma = 1.0
    result = basic_reproduction(beta, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_r0bayse_edge():
    """Test edge cases."""
    beta = 0.8
    gamma = 1.0
    result = basic_reproduction(beta, gamma)
    assert isinstance(result, dict)
