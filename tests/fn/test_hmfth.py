"""Tests for hmfth.geron_finetune_lm."""
import numpy as np
import pytest
from moirais.fn.hmfth import geron_finetune_lm


def test_hmfth_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    dataset = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_finetune_lm(model, dataset, epochs, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmfth_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    dataset = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_finetune_lm(model, dataset, epochs, lr)
    assert isinstance(result, dict)
