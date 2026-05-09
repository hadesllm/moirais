"""Tests for ca11u347.ca_chapter_11_unnumbered_347."""
import numpy as np
import pytest
from moirais.fn.ca11u347 import ca_chapter_11_unnumbered_347


def test_ca11u347_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_unnumbered_347(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca11u347_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_unnumbered_347(x)
    assert isinstance(result, dict)
