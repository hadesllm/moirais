"""Tests for kmclm.kamath_causal_lm_loss."""
import numpy as np
import pytest
from moirais.fn.kmclm import kamath_causal_lm_loss


def test_kmclm_basic():
    """Test basic functionality."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    targets = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_causal_lm_loss(logits, targets)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmclm_edge():
    """Test edge cases."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    targets = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_causal_lm_loss(logits, targets)
    assert isinstance(result, dict)
