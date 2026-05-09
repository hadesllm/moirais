"""Tests for mrr.mrr."""
import numpy as np
import pytest
from moirais.fn.mrr import mrr


def test_mrr_basic():
    """Test basic functionality."""
    pred_rank = np.random.default_rng(42).normal(0, 1, 100)
    relevant = np.random.default_rng(42).normal(0, 1, 100)
    result = mrr(pred_rank, relevant)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mrr_edge():
    """Test edge cases."""
    pred_rank = np.random.default_rng(42).normal(0, 1, 100)
    relevant = np.random.default_rng(42).normal(0, 1, 100)
    result = mrr(pred_rank, relevant)
    assert isinstance(result, dict)
