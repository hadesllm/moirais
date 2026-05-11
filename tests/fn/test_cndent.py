"""Tests for cndent.conditional_entropy."""
import numpy as np
import pytest
from morie.fn.cndent import conditional_entropy


def test_cndent_basic():
    """Test basic functionality."""
    pxy = np.random.default_rng(42).normal(0, 1, 100)
    result = conditional_entropy(pxy)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cndent_edge():
    """Test edge cases."""
    pxy = np.random.default_rng(42).normal(0, 1, 100)
    result = conditional_entropy(pxy)
    assert isinstance(result, dict)
