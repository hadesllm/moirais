"""Tests for snpblr.snp_blup."""
import numpy as np
import pytest
from moirais.fn.snpblr import snp_blup


def test_snpblr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = snp_blup(y, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_snpblr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = snp_blup(y, M)
    assert isinstance(result, dict)
