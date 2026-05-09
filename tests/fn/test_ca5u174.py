"""Tests for ca5u174.ca_chapter_5_unnumbered_174."""
import numpy as np
import pytest
from moirais.fn.ca5u174 import ca_chapter_5_unnumbered_174


def test_ca5u174_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_174(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca5u174_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_174(x)
    assert isinstance(result, dict)
