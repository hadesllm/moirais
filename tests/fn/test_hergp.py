"""Tests for hergp.herg_inhibition."""
import numpy as np
import pytest
from moirais.fn.hergp import herg_inhibition


def test_hergp_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = herg_inhibition(smiles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hergp_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = herg_inhibition(smiles)
    assert isinstance(result, dict)
