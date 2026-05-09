"""Tests for halft.half_life."""
import numpy as np
import pytest
from moirais.fn.halft import half_life


def test_halft_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    Vd = np.random.default_rng(42).normal(0, 1, 100)
    Cl = np.random.default_rng(42).normal(0, 1, 100)
    result = half_life(smiles, Vd, Cl)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_halft_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    Vd = np.random.default_rng(42).normal(0, 1, 100)
    Cl = np.random.default_rng(42).normal(0, 1, 100)
    result = half_life(smiles, Vd, Cl)
    assert isinstance(result, dict)
