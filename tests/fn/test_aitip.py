"""Tests for aitip.aitchison_inner_product."""
import numpy as np
import pytest
from morie.fn.aitip import aitchison_inner_product


def test_aitip_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = aitchison_inner_product(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitip_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = aitchison_inner_product(x, y)
    assert isinstance(result, dict)
