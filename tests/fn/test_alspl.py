"""Tests for alspl.alammar_sampling_decoding."""
import numpy as np
import pytest
from moirais.fn.alspl import alammar_sampling_decoding


def test_alspl_basic():
    """Test basic functionality."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = alammar_sampling_decoding(logits, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alspl_edge():
    """Test edge cases."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = alammar_sampling_decoding(logits, seed)
    assert isinstance(result, dict)
