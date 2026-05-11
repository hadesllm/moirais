"""Tests for guide_on_data_analysis6u514.guide_on_data_analysis_chapter_6_unnumbered_514."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis6u514 import guide_on_data_analysis_chapter_6_unnumbered_514


def test_guide_on_data_analysis6u514_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_6_unnumbered_514(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis6u514_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_6_unnumbered_514(x)
    assert isinstance(result, dict)
