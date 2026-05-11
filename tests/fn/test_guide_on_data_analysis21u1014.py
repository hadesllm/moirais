"""Tests for guide_on_data_analysis21u1014.guide_on_data_analysis_chapter_21_unnumbered_1014."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis21u1014 import guide_on_data_analysis_chapter_21_unnumbered_1014


def test_guide_on_data_analysis21u1014_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_21_unnumbered_1014(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis21u1014_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_21_unnumbered_1014(x)
    assert isinstance(result, dict)
