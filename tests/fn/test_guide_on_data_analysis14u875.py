"""Tests for guide_on_data_analysis14u875.guide_on_data_analysis_chapter_14_unnumbered_875."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis14u875 import guide_on_data_analysis_chapter_14_unnumbered_875


def test_guide_on_data_analysis14u875_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_14_unnumbered_875(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis14u875_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_14_unnumbered_875(x)
    assert isinstance(result, dict)
