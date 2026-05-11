"""Tests for sasc1.synthetic_accessibility."""
import numpy as np
import pytest
from morie.fn.sasc1 import synthetic_accessibility


def test_sasc1_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = synthetic_accessibility(smiles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sasc1_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = synthetic_accessibility(smiles)
    assert isinstance(result, dict)
