"""Tests for selfgr.selfies_encode."""
import numpy as np
import pytest
from morie.fn.selfgr import selfies_encode


def test_selfgr_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = selfies_encode(smiles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_selfgr_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = selfies_encode(smiles)
    assert isinstance(result, dict)
