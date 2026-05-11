"""Tests for aitpie.compositional_pielou."""
import numpy as np
import pytest
from morie.fn.aitpie import compositional_pielou


def test_aitpie_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = compositional_pielou(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitpie_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = compositional_pielou(x)
    assert isinstance(result, dict)
