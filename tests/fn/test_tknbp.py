"""Tests for tknbp.bpe_tokenizer."""
import numpy as np
import pytest
from morie.fn.tknbp import bpe_tokenizer


def test_tknbp_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = bpe_tokenizer(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_tknbp_edge():
    """Test edge cases."""
    result = bpe_tokenizer(np.array([42.0]))
    assert result['n'] == 1
