"""Tests for hmchrn.geron_char_rnn."""
import numpy as np
import pytest
from morie.fn.hmchrn import geron_char_rnn


def test_hmchrn_basic():
    """Test basic functionality."""
    text = np.random.default_rng(42).normal(0, 1, 100)
    hidden = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_char_rnn(text, hidden, epochs, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmchrn_edge():
    """Test edge cases."""
    text = np.random.default_rng(42).normal(0, 1, 100)
    hidden = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_char_rnn(text, hidden, epochs, lr)
    assert isinstance(result, dict)
