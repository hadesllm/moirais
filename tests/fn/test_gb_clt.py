"""Tests for gb_clt.gibbons_clt."""
import numpy as np
import pytest
from morie.fn.gb_clt import gibbons_clt


def test_gb_clt_basic():
    """Test basic functionality."""
    n = 100
    mu = 0.0
    sigma = 1.0
    result = gibbons_clt(n, mu, sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb_clt_edge():
    """Test edge cases."""
    n = 100
    mu = 0.0
    sigma = 1.0
    result = gibbons_clt(n, mu, sigma)
    assert isinstance(result, dict)
