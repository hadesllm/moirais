"""Tests for psar2.polar_surface_area."""
import numpy as np
import pytest
from moirais.fn.psar2 import polar_surface_area


def test_psar2_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = polar_surface_area(smiles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_psar2_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = polar_surface_area(smiles)
    assert isinstance(result, dict)
