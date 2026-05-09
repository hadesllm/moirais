"""Tests for ca4u127.ca_chapter_4_unnumbered_127."""
import numpy as np
import pytest
from moirais.fn.ca4u127 import ca_chapter_4_unnumbered_127


def test_ca4u127_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_127(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca4u127_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_127(x)
    assert isinstance(result, dict)
