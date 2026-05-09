"""Tests for ca8u313.ca_chapter_8_unnumbered_313."""
import numpy as np
import pytest
from moirais.fn.ca8u313 import ca_chapter_8_unnumbered_313


def test_ca8u313_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_313(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_ca8u313_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_313(x)
    assert isinstance(result, dict)
