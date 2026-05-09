"""Tests for aitdrs.dirichlet_sample."""
import numpy as np
import pytest
from moirais.fn.aitdrs import dirichlet_sample


def test_aitdrs_basic():
    """Test basic functionality."""
    alpha = 0.05
    n = 100
    result = dirichlet_sample(alpha, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitdrs_edge():
    """Test edge cases."""
    alpha = 0.05
    n = 100
    result = dirichlet_sample(alpha, n)
    assert isinstance(result, dict)
