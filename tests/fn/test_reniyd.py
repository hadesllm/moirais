"""Tests for reniyd.renyi_entropy."""
import numpy as np
import pytest
from moirais.fn.reniyd import renyi_entropy


def test_reniyd_basic():
    """Test basic functionality."""
    p = 5
    alpha = 0.05
    result = renyi_entropy(p, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_reniyd_edge():
    """Test edge cases."""
    p = 5
    alpha = 0.05
    result = renyi_entropy(p, alpha)
    assert isinstance(result, dict)
