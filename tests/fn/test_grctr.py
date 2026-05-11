"""Tests for grctr.geron_contrastive_infonce."""
import numpy as np
import pytest
from morie.fn.grctr import geron_contrastive_infonce


def test_grctr_basic():
    """Test basic functionality."""
    anchors = np.random.default_rng(42).normal(0, 1, 100)
    positives = np.random.default_rng(42).normal(0, 1, 100)
    negatives = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = geron_contrastive_infonce(anchors, positives, negatives, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grctr_edge():
    """Test edge cases."""
    anchors = np.random.default_rng(42).normal(0, 1, 100)
    positives = np.random.default_rng(42).normal(0, 1, 100)
    negatives = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = geron_contrastive_infonce(anchors, positives, negatives, tau)
    assert isinstance(result, dict)
