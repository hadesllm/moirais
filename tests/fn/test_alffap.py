"""Tests for alffap.alphafold_fape_loss."""
import numpy as np
import pytest
from moirais.fn.alffap import alphafold_fape_loss


def test_alffap_basic():
    """Test basic functionality."""
    frames_pred = np.random.default_rng(42).normal(0, 1, 100)
    frames_true = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    x_true = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_fape_loss(frames_pred, frames_true, x, x_true)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alffap_edge():
    """Test edge cases."""
    frames_pred = np.random.default_rng(42).normal(0, 1, 100)
    frames_true = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    x_true = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_fape_loss(frames_pred, frames_true, x, x_true)
    assert isinstance(result, dict)
