"""Tests for egan2.egan_filter."""
import numpy as np
import pytest
from morie.fn.egan2 import egan_filter


def test_egan2_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = egan_filter(smiles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_egan2_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = egan_filter(smiles)
    assert isinstance(result, dict)
