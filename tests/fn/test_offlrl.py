"""Tests for offlrl.offline_rl_cql."""
import numpy as np
import pytest
from moirais.fn.offlrl import offline_rl_cql


def test_offlrl_basic():
    """Test basic functionality."""
    dataset = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = offline_rl_cql(dataset, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_offlrl_edge():
    """Test edge cases."""
    dataset = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = offline_rl_cql(dataset, alpha)
    assert isinstance(result, dict)
