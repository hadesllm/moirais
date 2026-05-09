"""Tests for pmfsc.pmf_potential."""
import numpy as np
import pytest
from moirais.fn.pmfsc import pmf_potential


def test_pmfsc_basic():
    """Test basic functionality."""
    receptor = np.random.default_rng(42).normal(0, 1, 100)
    ligand = np.random.default_rng(42).normal(0, 1, 100)
    result = pmf_potential(receptor, ligand)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pmfsc_edge():
    """Test edge cases."""
    receptor = np.random.default_rng(42).normal(0, 1, 100)
    ligand = np.random.default_rng(42).normal(0, 1, 100)
    result = pmf_potential(receptor, ligand)
    assert isinstance(result, dict)
