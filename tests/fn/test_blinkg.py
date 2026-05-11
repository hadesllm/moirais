"""Tests for blinkg.blink_gwas."""
import numpy as np
import pytest
from morie.fn.blinkg import blink_gwas


def test_blinkg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = blink_gwas(y, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_blinkg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = blink_gwas(y, M)
    assert isinstance(result, dict)
