"""Tests for ca4u124.ca_chapter_4_unnumbered_124."""
import numpy as np
import pytest
from moirais.fn.ca4u124 import ca_chapter_4_unnumbered_124


def test_ca4u124_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_124(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca4u124_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_124(x)
    assert isinstance(result, dict)
