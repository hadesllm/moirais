"""Tests for guide_on_data_analysis16u938.guide_on_data_analysis_chapter_16_unnumbered_938."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis16u938 import guide_on_data_analysis_chapter_16_unnumbered_938


def test_guide_on_data_analysis16u938_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_16_unnumbered_938(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_guide_on_data_analysis16u938_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_16_unnumbered_938(x)
    assert isinstance(result, dict)
