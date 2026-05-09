"""Tests for aitjsd.compositional_jsd."""
import numpy as np
import pytest
from moirais.fn.aitjsd import compositional_jsd


def test_aitjsd_basic():
    """Test basic functionality."""
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = compositional_jsd(p, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitjsd_edge():
    """Test edge cases."""
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = compositional_jsd(p, q)
    assert isinstance(result, dict)
