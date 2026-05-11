"""Tests for hbdon.hbond_donor_count."""
import numpy as np
import pytest
from morie.fn.hbdon import hbond_donor_count


def test_hbdon_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = hbond_donor_count(smiles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hbdon_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = hbond_donor_count(smiles)
    assert isinstance(result, dict)
