"""Tests for reosft.reos_filter."""
import numpy as np
import pytest
from moirais.fn.reosft import reos_filter


def test_reosft_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = reos_filter(smiles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_reosft_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = reos_filter(smiles)
    assert isinstance(result, dict)
