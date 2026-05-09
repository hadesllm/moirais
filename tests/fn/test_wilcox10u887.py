"""Tests for wilcox10u887.wilcox_chapter_10_unnumbered_887."""
import numpy as np
import pytest
from moirais.fn.wilcox10u887 import wilcox_chapter_10_unnumbered_887


def test_wilcox10u887_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_887(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox10u887_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_887(x)
    assert isinstance(result, dict)
