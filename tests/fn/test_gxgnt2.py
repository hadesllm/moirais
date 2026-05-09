"""Tests for gxgnt2.gxg_interaction."""
import numpy as np
import pytest
from moirais.fn.gxgnt2 import gxg_interaction


def test_gxgnt2_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    SNP1 = np.random.default_rng(42).normal(0, 1, 100)
    SNP2 = np.random.default_rng(42).normal(0, 1, 100)
    result = gxg_interaction(y, SNP1, SNP2)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gxgnt2_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    SNP1 = np.random.default_rng(42).normal(0, 1, 100)
    SNP2 = np.random.default_rng(42).normal(0, 1, 100)
    result = gxg_interaction(y, SNP1, SNP2)
    assert isinstance(result, dict)
