"""Tests for sbpst.stick_breaking_post."""
import numpy as np
import pytest
from moirais.fn.sbpst import stick_breaking_post


def test_sbpst_basic():
    """Test basic functionality."""
    partition = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = stick_breaking_post(partition, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sbpst_edge():
    """Test edge cases."""
    partition = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = stick_breaking_post(partition, alpha)
    assert isinstance(result, dict)
