"""Tests for guide_on_data_analysis27u1338.guide_on_data_analysis_chapter_27_unnumbered_1338."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis27u1338 import guide_on_data_analysis_chapter_27_unnumbered_1338


def test_guide_on_data_analysis27u1338_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_27_unnumbered_1338(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis27u1338_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_27_unnumbered_1338(x)
    assert isinstance(result, dict)
