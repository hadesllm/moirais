"""Tests for trfbl.transformer_block."""
import numpy as np
import pytest
from moirais.fn.trfbl import transformer_block


def test_trfbl_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = transformer_block(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_trfbl_edge():
    """Test edge cases."""
    result = transformer_block(np.array([42.0]))
    assert result['n'] == 1
