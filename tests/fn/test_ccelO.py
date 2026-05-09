"""Tests for ccelO.categorical_crossentropy_loss."""
import numpy as np
import pytest
from moirais.fn.ccelO import categorical_crossentropy_loss


def test_ccelO_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    result = categorical_crossentropy_loss(y, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ccelO_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    result = categorical_crossentropy_loss(y, p)
    assert isinstance(result, dict)
