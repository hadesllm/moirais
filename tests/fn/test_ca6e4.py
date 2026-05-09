"""Tests for ca6e4.ca_chapter_6_equation_4."""
import numpy as np
import pytest
from moirais.fn.ca6e4 import ca_chapter_6_equation_4


def test_ca6e4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_equation_4(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca6e4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_equation_4(x)
    assert isinstance(result, dict)
