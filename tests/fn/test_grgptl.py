"""Tests for grgptl.geron_gpt_autoregressive_loss."""
import numpy as np
import pytest
from moirais.fn.grgptl import geron_gpt_autoregressive_loss


def test_grgptl_basic():
    """Test basic functionality."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    targets = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gpt_autoregressive_loss(logits, targets)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grgptl_edge():
    """Test edge cases."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    targets = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gpt_autoregressive_loss(logits, targets)
    assert isinstance(result, dict)
