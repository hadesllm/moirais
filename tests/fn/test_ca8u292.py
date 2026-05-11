"""Tests for ca8u292.ca_chapter_8_unnumbered_292."""
import numpy as np
import pytest
from morie.fn.ca8u292 import ca_chapter_8_unnumbered_292


def test_ca8u292_basic():
    """Test basic functionality."""
    x1 = np.random.default_rng(42).normal(0, 1, 100)
    x2 = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_292(x1, x2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca8u292_edge():
    """Test edge cases."""
    x1 = np.random.default_rng(42).normal(0, 1, 100)
    x2 = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_292(x1, x2)
    assert isinstance(result, dict)
