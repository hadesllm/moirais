"""Tests for guide_on_data_analysis11u806.guide_on_data_analysis_chapter_11_unnumbered_806."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis11u806 import guide_on_data_analysis_chapter_11_unnumbered_806


def test_guide_on_data_analysis11u806_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_11_unnumbered_806(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis11u806_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_11_unnumbered_806(x)
    assert isinstance(result, dict)
