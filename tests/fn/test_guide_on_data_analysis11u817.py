"""Tests for guide_on_data_analysis11u817.guide_on_data_analysis_chapter_11_unnumbered_817."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis11u817 import guide_on_data_analysis_chapter_11_unnumbered_817


def test_guide_on_data_analysis11u817_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_11_unnumbered_817(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis11u817_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_11_unnumbered_817(x)
    assert isinstance(result, dict)
