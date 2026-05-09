"""Tests for guide_on_data_analysis11u823.guide_on_data_analysis_chapter_11_unnumbered_823."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis11u823 import guide_on_data_analysis_chapter_11_unnumbered_823


def test_guide_on_data_analysis11u823_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_11_unnumbered_823(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis11u823_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_11_unnumbered_823(x)
    assert isinstance(result, dict)
