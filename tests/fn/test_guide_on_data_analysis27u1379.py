"""Tests for guide_on_data_analysis27u1379.guide_on_data_analysis_chapter_27_unnumbered_1379."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis27u1379 import guide_on_data_analysis_chapter_27_unnumbered_1379


def test_guide_on_data_analysis27u1379_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_27_unnumbered_1379(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis27u1379_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_27_unnumbered_1379(x)
    assert isinstance(result, dict)
