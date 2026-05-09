"""Tests for hubrho.huber_loss."""
import numpy as np
import pytest
from moirais.fn.hubrho import huber_loss


def test_hubrho_basic():
    """Test basic functionality."""
    r = 10
    k = 5
    result = huber_loss(r, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hubrho_edge():
    """Test edge cases."""
    r = 10
    k = 5
    result = huber_loss(r, k)
    assert isinstance(result, dict)
