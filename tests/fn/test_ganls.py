"""Tests for ganls.gan_loss."""
import numpy as np
import pytest
from moirais.fn.ganls import gan_loss


def test_ganls_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gan_loss(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ganls_edge():
    """Test edge cases."""
    result = gan_loss(np.array([42.0]))
    assert result['n'] == 1
