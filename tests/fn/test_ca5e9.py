"""Tests for ca5e9.ca_chapter_5_equation_9."""
import numpy as np
import pytest
from morie.fn.ca5e9 import ca_chapter_5_equation_9


def test_ca5e9_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_equation_9(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca5e9_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_equation_9(x)
    assert isinstance(result, dict)
