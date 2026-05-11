"""Tests for fanocb.fano_inequality."""
import numpy as np
import pytest
from morie.fn.fanocb import fano_inequality


def test_fanocb_basic():
    """Test basic functionality."""
    pe = np.random.default_rng(42).normal(0, 1, 100)
    X_card = np.random.default_rng(42).normal(0, 1, 100)
    result = fano_inequality(pe, X_card)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fanocb_edge():
    """Test edge cases."""
    pe = np.random.default_rng(42).normal(0, 1, 100)
    X_card = np.random.default_rng(42).normal(0, 1, 100)
    result = fano_inequality(pe, X_card)
    assert isinstance(result, dict)
