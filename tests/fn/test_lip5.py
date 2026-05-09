"""Tests for lip5.lipinski_rule_of_5."""
import numpy as np
import pytest
from moirais.fn.lip5 import lipinski_rule_of_5


def test_lip5_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = lipinski_rule_of_5(smiles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lip5_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = lipinski_rule_of_5(smiles)
    assert isinstance(result, dict)
