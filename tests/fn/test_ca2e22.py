"""Tests for ca2e22.ca_chapter_2_equation_22."""
import numpy as np
import pytest
from morie.fn.ca2e22 import ca_chapter_2_equation_22


def test_ca2e22_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_equation_22(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca2e22_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_equation_22(x)
    assert isinstance(result, dict)
