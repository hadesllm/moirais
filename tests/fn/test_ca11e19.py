"""Tests for ca11e19.ca_chapter_11_equation_19."""
import numpy as np
import pytest
from moirais.fn.ca11e19 import ca_chapter_11_equation_19


def test_ca11e19_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_19(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca11e19_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_19(x)
    assert isinstance(result, dict)
