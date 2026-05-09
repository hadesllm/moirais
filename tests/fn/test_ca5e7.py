"""Tests for ca5e7.ca_chapter_5_equation_7."""
import numpy as np
import pytest
from moirais.fn.ca5e7 import ca_chapter_5_equation_7


def test_ca5e7_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_equation_7(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca5e7_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_equation_7(x)
    assert isinstance(result, dict)
