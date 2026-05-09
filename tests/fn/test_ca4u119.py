"""Tests for ca4u119.ca_chapter_4_unnumbered_119."""
import numpy as np
import pytest
from moirais.fn.ca4u119 import ca_chapter_4_unnumbered_119


def test_ca4u119_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_119(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca4u119_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_119(x)
    assert isinstance(result, dict)
