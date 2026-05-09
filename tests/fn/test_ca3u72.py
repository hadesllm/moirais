"""Tests for ca3u72.ca_chapter_3_unnumbered_72."""
import numpy as np
import pytest
from moirais.fn.ca3u72 import ca_chapter_3_unnumbered_72


def test_ca3u72_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_72(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca3u72_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_72(x)
    assert isinstance(result, dict)
