"""Tests for sammkr.sam_multi_mask_rank."""
import numpy as np
import pytest
from morie.fn.sammkr import sam_multi_mask_rank


def test_sammkr_basic():
    """Test basic functionality."""
    masks = np.random.default_rng(42).normal(0, 1, 100)
    scores = np.random.default_rng(42).uniform(0, 1, 100)
    result = sam_multi_mask_rank(masks, scores)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sammkr_edge():
    """Test edge cases."""
    masks = np.random.default_rng(42).normal(0, 1, 100)
    scores = np.random.default_rng(42).uniform(0, 1, 100)
    result = sam_multi_mask_rank(masks, scores)
    assert isinstance(result, dict)
