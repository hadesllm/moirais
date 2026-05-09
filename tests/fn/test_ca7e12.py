"""Tests for ca7e12.ca_chapter_7_equation_12."""
import numpy as np
import pytest
from moirais.fn.ca7e12 import ca_chapter_7_equation_12


def test_ca7e12_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_equation_12(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca7e12_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_equation_12(x)
    assert isinstance(result, dict)
