"""Tests for guide_on_data_analysis4u213.guide_on_data_analysis_chapter_4_unnumbered_213."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis4u213 import guide_on_data_analysis_chapter_4_unnumbered_213


def test_guide_on_data_analysis4u213_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_4_unnumbered_213(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis4u213_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_4_unnumbered_213(x)
    assert isinstance(result, dict)
