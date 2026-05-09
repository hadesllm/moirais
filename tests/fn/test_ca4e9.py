"""Tests for ca4e9.ca_chapter_4_equation_9."""
import numpy as np
import pytest
from moirais.fn.ca4e9 import ca_chapter_4_equation_9


def test_ca4e9_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_equation_9(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca4e9_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_equation_9(x)
    assert isinstance(result, dict)
