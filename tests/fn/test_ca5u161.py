"""Tests for ca5u161.ca_chapter_5_unnumbered_161."""
import numpy as np
import pytest
from moirais.fn.ca5u161 import ca_chapter_5_unnumbered_161


def test_ca5u161_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_161(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca5u161_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_161(x)
    assert isinstance(result, dict)
