"""Tests for ca6e6.ca_chapter_6_equation_6."""
import numpy as np
import pytest
from morie.fn.ca6e6 import ca_chapter_6_equation_6


def test_ca6e6_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_equation_6(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca6e6_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_equation_6(x)
    assert isinstance(result, dict)
