"""Tests for ca2e4.ca_chapter_2_equation_4."""
import numpy as np
import pytest
from moirais.fn.ca2e4 import ca_chapter_2_equation_4


def test_ca2e4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_equation_4(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca2e4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_equation_4(x)
    assert isinstance(result, dict)
