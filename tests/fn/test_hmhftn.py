"""Tests for hmhftn.geron_hf_trainer."""
import numpy as np
import pytest
from moirais.fn.hmhftn import geron_hf_trainer


def test_hmhftn_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    args = np.random.default_rng(42).normal(0, 1, 100)
    train_ds = np.random.default_rng(42).normal(0, 1, 100)
    eval_ds = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_hf_trainer(model, args, train_ds, eval_ds)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmhftn_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    args = np.random.default_rng(42).normal(0, 1, 100)
    train_ds = np.random.default_rng(42).normal(0, 1, 100)
    eval_ds = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_hf_trainer(model, args, train_ds, eval_ds)
    assert isinstance(result, dict)
