"""Tests for grmlm.geron_bert_mlm_loss."""
import numpy as np
import pytest
from morie.fn.grmlm import geron_bert_mlm_loss


def test_grmlm_basic():
    """Test basic functionality."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    targets = np.random.default_rng(42).normal(0, 1, 100)
    mask = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bert_mlm_loss(logits, targets, mask)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grmlm_edge():
    """Test edge cases."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    targets = np.random.default_rng(42).normal(0, 1, 100)
    mask = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bert_mlm_loss(logits, targets, mask)
    assert isinstance(result, dict)
