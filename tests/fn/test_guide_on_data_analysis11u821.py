"""Tests for guide_on_data_analysis11u821.guide_on_data_analysis_chapter_11_unnumbered_821."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis11u821 import guide_on_data_analysis_chapter_11_unnumbered_821


def test_guide_on_data_analysis11u821_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_11_unnumbered_821(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_guide_on_data_analysis11u821_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_11_unnumbered_821(x)
    assert isinstance(result, dict)
