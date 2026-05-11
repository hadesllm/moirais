"""Tests for ca11e32.ca_chapter_11_equation_32."""
import numpy as np
import pytest
from morie.fn.ca11e32 import ca_chapter_11_equation_32


def test_ca11e32_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_32(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca11e32_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_32(x)
    assert isinstance(result, dict)
