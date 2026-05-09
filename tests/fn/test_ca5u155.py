"""Tests for ca5u155.ca_chapter_5_unnumbered_155."""
import numpy as np
import pytest
from moirais.fn.ca5u155 import ca_chapter_5_unnumbered_155


def test_ca5u155_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_155(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca5u155_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_155(x)
    assert isinstance(result, dict)
