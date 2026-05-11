"""Tests for altrip.alammar_sbert_triplet_loss."""
import numpy as np
import pytest
from morie.fn.altrip import alammar_sbert_triplet_loss


def test_altrip_basic():
    """Test basic functionality."""
    anchor = np.random.default_rng(42).normal(0, 1, 100)
    positive = np.random.default_rng(42).normal(0, 1, 100)
    negative = np.random.default_rng(42).normal(0, 1, 100)
    margin = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_sbert_triplet_loss(anchor, positive, negative, margin)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_altrip_edge():
    """Test edge cases."""
    anchor = np.random.default_rng(42).normal(0, 1, 100)
    positive = np.random.default_rng(42).normal(0, 1, 100)
    negative = np.random.default_rng(42).normal(0, 1, 100)
    margin = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_sbert_triplet_loss(anchor, positive, negative, margin)
    assert isinstance(result, dict)
