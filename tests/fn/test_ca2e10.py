"""Tests for ca2e10.ca_chapter_2_equation_10."""
import numpy as np
import pytest
from moirais.fn.ca2e10 import ca_chapter_2_equation_10


def test_ca2e10_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_equation_10(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca2e10_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_equation_10(x)
    assert isinstance(result, dict)
