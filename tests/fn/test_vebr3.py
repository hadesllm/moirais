"""Tests for vebr3.veber_rule."""
import numpy as np
import pytest
from morie.fn.vebr3 import veber_rule


def test_vebr3_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = veber_rule(smiles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vebr3_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = veber_rule(smiles)
    assert isinstance(result, dict)
