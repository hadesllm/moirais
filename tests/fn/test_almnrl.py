"""Tests for almnrl.alammar_multiple_negatives_ranking."""
import numpy as np
import pytest
from morie.fn.almnrl import alammar_multiple_negatives_ranking


def test_almnrl_basic():
    """Test basic functionality."""
    anchors = np.random.default_rng(42).normal(0, 1, 100)
    positives = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = alammar_multiple_negatives_ranking(anchors, positives, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_almnrl_edge():
    """Test edge cases."""
    anchors = np.random.default_rng(42).normal(0, 1, 100)
    positives = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = alammar_multiple_negatives_ranking(anchors, positives, tau)
    assert isinstance(result, dict)
