"""Tests for bbbpr.bbb_permeability."""
import numpy as np
import pytest
from moirais.fn.bbbpr import bbb_permeability


def test_bbbpr_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = bbb_permeability(smiles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bbbpr_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = bbb_permeability(smiles)
    assert isinstance(result, dict)
