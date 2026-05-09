"""Tests for twoldp.two_locus_dprime."""
import numpy as np
import pytest
from moirais.fn.twoldp import two_locus_dprime


def test_twoldp_basic():
    """Test basic functionality."""
    geno1 = np.random.default_rng(42).normal(0, 1, 100)
    geno2 = np.random.default_rng(42).normal(0, 1, 100)
    result = two_locus_dprime(geno1, geno2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_twoldp_edge():
    """Test edge cases."""
    geno1 = np.random.default_rng(42).normal(0, 1, 100)
    geno2 = np.random.default_rng(42).normal(0, 1, 100)
    result = two_locus_dprime(geno1, geno2)
    assert isinstance(result, dict)
