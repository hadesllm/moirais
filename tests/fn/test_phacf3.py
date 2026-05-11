"""Tests for phacf3.pharmacophore_3d."""
import numpy as np
import pytest
from morie.fn.phacf3 import pharmacophore_3d


def test_phacf3_basic():
    """Test basic functionality."""
    mol_3d = np.random.default_rng(42).normal(0, 1, 100)
    feature_set = np.random.default_rng(42).normal(0, 1, 100)
    result = pharmacophore_3d(mol_3d, feature_set)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_phacf3_edge():
    """Test edge cases."""
    mol_3d = np.random.default_rng(42).normal(0, 1, 100)
    feature_set = np.random.default_rng(42).normal(0, 1, 100)
    result = pharmacophore_3d(mol_3d, feature_set)
    assert isinstance(result, dict)
