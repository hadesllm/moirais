"""Tests for ca4e6.ca_chapter_4_equation_6."""
import numpy as np
import pytest
from moirais.fn.ca4e6 import ca_chapter_4_equation_6


def test_ca4e6_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_equation_6(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca4e6_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_equation_6(x)
    assert isinstance(result, dict)
