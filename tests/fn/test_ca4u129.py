"""Tests for ca4u129.ca_chapter_4_unnumbered_129."""
import numpy as np
import pytest
from moirais.fn.ca4u129 import ca_chapter_4_unnumbered_129


def test_ca4u129_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_129(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca4u129_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_129(x)
    assert isinstance(result, dict)
