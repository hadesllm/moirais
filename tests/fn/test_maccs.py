"""Tests for maccs.maccs_keys."""
import numpy as np
import pytest
from morie.fn.maccs import maccs_keys


def test_maccs_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = maccs_keys(smiles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_maccs_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = maccs_keys(smiles)
    assert isinstance(result, dict)
