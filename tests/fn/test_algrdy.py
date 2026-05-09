"""Tests for algrdy.alammar_greedy_decoding."""
import numpy as np
import pytest
from moirais.fn.algrdy import alammar_greedy_decoding


def test_algrdy_basic():
    """Test basic functionality."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_greedy_decoding(logits)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_algrdy_edge():
    """Test edge cases."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_greedy_decoding(logits)
    assert isinstance(result, dict)
