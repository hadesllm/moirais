"""Tests for qedscr.qed_drug_likeness."""
import numpy as np
import pytest
from morie.fn.qedscr import qed_drug_likeness


def test_qedscr_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = qed_drug_likeness(smiles, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_qedscr_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = qed_drug_likeness(smiles, weights)
    assert isinstance(result, dict)
