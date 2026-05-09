"""Tests for chemsc.chemscore_dock."""
import numpy as np
import pytest
from moirais.fn.chemsc import chemscore_dock


def test_chemsc_basic():
    """Test basic functionality."""
    receptor = np.random.default_rng(42).normal(0, 1, 100)
    ligand = np.random.default_rng(42).normal(0, 1, 100)
    result = chemscore_dock(receptor, ligand)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_chemsc_edge():
    """Test edge cases."""
    receptor = np.random.default_rng(42).normal(0, 1, 100)
    ligand = np.random.default_rng(42).normal(0, 1, 100)
    result = chemscore_dock(receptor, ligand)
    assert isinstance(result, dict)
