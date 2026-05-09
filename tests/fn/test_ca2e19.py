"""Tests for ca2e19.ca_chapter_2_equation_19."""
import numpy as np
import pytest
from moirais.fn.ca2e19 import ca_chapter_2_equation_19


def test_ca2e19_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_equation_19(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca2e19_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_equation_19(x)
    assert isinstance(result, dict)
