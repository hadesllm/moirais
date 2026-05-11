"""Tests for cypin.cyp450_inhibition."""
import numpy as np
import pytest
from morie.fn.cypin import cyp450_inhibition


def test_cypin_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    isozyme = np.random.default_rng(42).normal(0, 1, 100)
    result = cyp450_inhibition(smiles, isozyme)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cypin_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    isozyme = np.random.default_rng(42).normal(0, 1, 100)
    result = cyp450_inhibition(smiles, isozyme)
    assert isinstance(result, dict)
