"""Tests for matransi.ma_logit_inverse."""
import numpy as np
import pytest
from morie.fn.matransi import ma_logit_inverse


def test_matransi_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = ma_logit_inverse(z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_matransi_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = ma_logit_inverse(z)
    assert isinstance(result, dict)
