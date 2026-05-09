"""Tests for alfrcl.alphafold_recycle_loss."""
import numpy as np
import pytest
from moirais.fn.alfrcl import alphafold_recycle_loss


def test_alfrcl_basic():
    """Test basic functionality."""
    frames_pred = np.random.default_rng(42).normal(0, 1, 100)
    frames_true = np.random.default_rng(42).normal(0, 1, 100)
    clamp = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_recycle_loss(frames_pred, frames_true, clamp)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alfrcl_edge():
    """Test edge cases."""
    frames_pred = np.random.default_rng(42).normal(0, 1, 100)
    frames_true = np.random.default_rng(42).normal(0, 1, 100)
    clamp = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_recycle_loss(frames_pred, frames_true, clamp)
    assert isinstance(result, dict)
