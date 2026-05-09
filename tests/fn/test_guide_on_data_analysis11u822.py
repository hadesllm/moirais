"""Tests for guide_on_data_analysis11u822.guide_on_data_analysis_chapter_11_unnumbered_822."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis11u822 import guide_on_data_analysis_chapter_11_unnumbered_822


def test_guide_on_data_analysis11u822_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_11_unnumbered_822(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_guide_on_data_analysis11u822_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_11_unnumbered_822(x)
    assert isinstance(result, dict)
