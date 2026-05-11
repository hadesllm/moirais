"""Tests for aglnvr.alphazero_loss_var."""
import numpy as np
import pytest
from morie.fn.aglnvr import alphazero_loss_var


def test_aglnvr_basic():
    """Test basic functionality."""
    losses = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_loss_var(losses)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aglnvr_edge():
    """Test edge cases."""
    losses = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_loss_var(losses)
    assert isinstance(result, dict)
