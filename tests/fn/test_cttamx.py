"""Tests for cttamx.ctt_alpha_max."""
import numpy as np
import pytest
from moirais.fn.cttamx import ctt_alpha_max


def test_cttamx_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = ctt_alpha_max(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cttamx_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = ctt_alpha_max(X)
    assert isinstance(result, dict)
