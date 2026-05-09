"""Tests for ames3.ames_mutagenicity."""
import numpy as np
import pytest
from moirais.fn.ames3 import ames_mutagenicity


def test_ames3_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = ames_mutagenicity(smiles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ames3_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = ames_mutagenicity(smiles)
    assert isinstance(result, dict)
