"""Tests for mwght.molecular_weight."""
import numpy as np
import pytest
from moirais.fn.mwght import molecular_weight


def test_mwght_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = molecular_weight(smiles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mwght_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = molecular_weight(smiles)
    assert isinstance(result, dict)
