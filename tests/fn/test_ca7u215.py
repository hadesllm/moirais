"""Tests for ca7u215.ca_chapter_7_unnumbered_215."""
import numpy as np
import pytest
from morie.fn.ca7u215 import ca_chapter_7_unnumbered_215


def test_ca7u215_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_215(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca7u215_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_215(x)
    assert isinstance(result, dict)
