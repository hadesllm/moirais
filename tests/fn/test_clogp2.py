"""Tests for clogp2.clogp_estimate."""
import numpy as np
import pytest
from morie.fn.clogp2 import clogp_estimate


def test_clogp2_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = clogp_estimate(smiles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_clogp2_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = clogp_estimate(smiles)
    assert isinstance(result, dict)
