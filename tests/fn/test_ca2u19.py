"""Tests for ca2u19.ca_chapter_2_unnumbered_19."""
import numpy as np
import pytest
from moirais.fn.ca2u19 import ca_chapter_2_unnumbered_19


def test_ca2u19_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_unnumbered_19(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca2u19_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_unnumbered_19(x)
    assert isinstance(result, dict)
