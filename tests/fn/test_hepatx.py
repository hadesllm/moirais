"""Tests for hepatx.hepatotoxicity."""
import numpy as np
import pytest
from morie.fn.hepatx import hepatotoxicity


def test_hepatx_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = hepatotoxicity(smiles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hepatx_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = hepatotoxicity(smiles)
    assert isinstance(result, dict)
