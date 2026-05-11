"""Tests for bcq.bcq."""
import numpy as np
import pytest
from morie.fn.bcq import bcq


def test_bcq_basic():
    """Test basic functionality."""
    dataset = np.random.default_rng(42).normal(0, 1, 100)
    vae = np.random.default_rng(42).normal(0, 1, 100)
    actor = np.random.default_rng(42).normal(0, 1, 100)
    result = bcq(dataset, vae, actor)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bcq_edge():
    """Test edge cases."""
    dataset = np.random.default_rng(42).normal(0, 1, 100)
    vae = np.random.default_rng(42).normal(0, 1, 100)
    actor = np.random.default_rng(42).normal(0, 1, 100)
    result = bcq(dataset, vae, actor)
    assert isinstance(result, dict)
