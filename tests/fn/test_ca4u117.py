"""Tests for ca4u117.ca_chapter_4_unnumbered_117."""
import numpy as np
import pytest
from morie.fn.ca4u117 import ca_chapter_4_unnumbered_117


def test_ca4u117_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_117(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca4u117_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_117(x)
    assert isinstance(result, dict)
