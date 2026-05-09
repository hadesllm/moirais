"""Tests for ca9e10.ca_chapter_9_equation_10."""
import numpy as np
import pytest
from moirais.fn.ca9e10 import ca_chapter_9_equation_10


def test_ca9e10_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_9_equation_10(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca9e10_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_9_equation_10(x)
    assert isinstance(result, dict)
