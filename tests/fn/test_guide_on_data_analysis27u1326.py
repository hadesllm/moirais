"""Tests for guide_on_data_analysis27u1326.guide_on_data_analysis_chapter_27_unnumbered_1326."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis27u1326 import guide_on_data_analysis_chapter_27_unnumbered_1326


def test_guide_on_data_analysis27u1326_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_27_unnumbered_1326(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis27u1326_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_27_unnumbered_1326(x)
    assert isinstance(result, dict)
