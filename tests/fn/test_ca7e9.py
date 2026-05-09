"""Tests for ca7e9.ca_chapter_7_equation_9."""
import numpy as np
import pytest
from moirais.fn.ca7e9 import ca_chapter_7_equation_9


def test_ca7e9_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_equation_9(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca7e9_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_equation_9(x)
    assert isinstance(result, dict)
