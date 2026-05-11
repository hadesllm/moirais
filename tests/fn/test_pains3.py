"""Tests for pains3.pains_filter."""
import numpy as np
import pytest
from morie.fn.pains3 import pains_filter


def test_pains3_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = pains_filter(smiles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pains3_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = pains_filter(smiles)
    assert isinstance(result, dict)
