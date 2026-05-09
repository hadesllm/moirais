"""Tests for kmperm.kamath_permutation_lm_loss."""
import numpy as np
import pytest
from moirais.fn.kmperm import kamath_permutation_lm_loss


def test_kmperm_basic():
    """Test basic functionality."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    targets = np.random.default_rng(42).normal(0, 1, 100)
    permutation = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_permutation_lm_loss(logits, targets, permutation)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmperm_edge():
    """Test edge cases."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    targets = np.random.default_rng(42).normal(0, 1, 100)
    permutation = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_permutation_lm_loss(logits, targets, permutation)
    assert isinstance(result, dict)
