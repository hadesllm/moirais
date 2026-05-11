"""Tests for hmbftn.geron_bert_finetune."""
import numpy as np
import pytest
from morie.fn.hmbftn import geron_bert_finetune


def test_hmbftn_basic():
    """Test basic functionality."""
    bert = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bert_finetune(bert, X, y, epochs, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmbftn_edge():
    """Test edge cases."""
    bert = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bert_finetune(bert, X, y, epochs, lr)
    assert isinstance(result, dict)
