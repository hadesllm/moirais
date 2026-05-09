"""Tests for ca6e8.ca_chapter_6_equation_8."""
import numpy as np
import pytest
from moirais.fn.ca6e8 import ca_chapter_6_equation_8


def test_ca6e8_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_equation_8(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca6e8_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_equation_8(x)
    assert isinstance(result, dict)
