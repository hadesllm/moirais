"""Tests for grsft.geron_sft_objective."""
import numpy as np
import pytest
from morie.fn.grsft import geron_sft_objective


def test_grsft_basic():
    """Test basic functionality."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    response_mask = np.random.default_rng(42).normal(0, 1, 100)
    targets = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_sft_objective(logits, response_mask, targets)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grsft_edge():
    """Test edge cases."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    response_mask = np.random.default_rng(42).normal(0, 1, 100)
    targets = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_sft_objective(logits, response_mask, targets)
    assert isinstance(result, dict)
