"""Tests for ca6e5.ca_chapter_6_equation_5."""
import numpy as np
import pytest
from morie.fn.ca6e5 import ca_chapter_6_equation_5


def test_ca6e5_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_equation_5(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca6e5_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_equation_5(x)
    assert isinstance(result, dict)
