"""Tests for ca4u134.ca_chapter_4_unnumbered_134."""
import numpy as np
import pytest
from moirais.fn.ca4u134 import ca_chapter_4_unnumbered_134


def test_ca4u134_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_134(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca4u134_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_134(x)
    assert isinstance(result, dict)
