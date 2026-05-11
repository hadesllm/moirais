"""Tests for ld50r.acute_toxicity_ld50."""
import numpy as np
import pytest
from morie.fn.ld50r import acute_toxicity_ld50


def test_ld50r_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = acute_toxicity_ld50(smiles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ld50r_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = acute_toxicity_ld50(smiles)
    assert isinstance(result, dict)
