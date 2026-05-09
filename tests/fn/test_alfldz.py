"""Tests for alfldz.alphafold_loss_decomposition."""
import numpy as np
import pytest
from moirais.fn.alfldz import alphafold_loss_decomposition


def test_alfldz_basic():
    """Test basic functionality."""
    loss_components = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_loss_decomposition(loss_components)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alfldz_edge():
    """Test edge cases."""
    loss_components = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_loss_decomposition(loss_components)
    assert isinstance(result, dict)
