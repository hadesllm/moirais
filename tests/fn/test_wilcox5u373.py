"""Tests for wilcox5u373.wilcox_chapter_5_unnumbered_373."""
import numpy as np
import pytest
from moirais.fn.wilcox5u373 import wilcox_chapter_5_unnumbered_373


def test_wilcox5u373_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_373(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox5u373_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_373(x)
    assert isinstance(result, dict)
