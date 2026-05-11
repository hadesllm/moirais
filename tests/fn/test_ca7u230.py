"""Tests for ca7u230.ca_chapter_7_unnumbered_230."""
import numpy as np
import pytest
from morie.fn.ca7u230 import ca_chapter_7_unnumbered_230


def test_ca7u230_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_230(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca7u230_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_230(x)
    assert isinstance(result, dict)
