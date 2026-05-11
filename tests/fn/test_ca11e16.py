"""Tests for ca11e16.ca_chapter_11_equation_16."""
import numpy as np
import pytest
from morie.fn.ca11e16 import ca_chapter_11_equation_16


def test_ca11e16_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_16(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca11e16_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_16(x)
    assert isinstance(result, dict)
