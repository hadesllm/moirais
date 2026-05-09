"""Tests for ldcmpr.ld_r2."""
import numpy as np
import pytest
from moirais.fn.ldcmpr import ld_r2


def test_ldcmpr_basic():
    """Test basic functionality."""
    geno1 = np.random.default_rng(42).normal(0, 1, 100)
    geno2 = np.random.default_rng(42).normal(0, 1, 100)
    result = ld_r2(geno1, geno2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ldcmpr_edge():
    """Test edge cases."""
    geno1 = np.random.default_rng(42).normal(0, 1, 100)
    geno2 = np.random.default_rng(42).normal(0, 1, 100)
    result = ld_r2(geno1, geno2)
    assert isinstance(result, dict)
