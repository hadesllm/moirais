"""Tests for hbacc.hbond_acceptor_count."""
import numpy as np
import pytest
from morie.fn.hbacc import hbond_acceptor_count


def test_hbacc_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = hbond_acceptor_count(smiles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hbacc_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = hbond_acceptor_count(smiles)
    assert isinstance(result, dict)
