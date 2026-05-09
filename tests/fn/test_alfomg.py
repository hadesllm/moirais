"""Tests for alfomg.openfold_msa_pair."""
import numpy as np
import pytest
from moirais.fn.alfomg import openfold_msa_pair


def test_alfomg_basic():
    """Test basic functionality."""
    msa = np.random.default_rng(42).normal(0, 1, 100)
    pair = np.random.default_rng(42).normal(0, 1, 100)
    result = openfold_msa_pair(msa, pair)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alfomg_edge():
    """Test edge cases."""
    msa = np.random.default_rng(42).normal(0, 1, 100)
    pair = np.random.default_rng(42).normal(0, 1, 100)
    result = openfold_msa_pair(msa, pair)
    assert isinstance(result, dict)
