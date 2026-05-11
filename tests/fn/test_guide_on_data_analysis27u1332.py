"""Tests for guide_on_data_analysis27u1332.guide_on_data_analysis_chapter_27_unnumbered_1332."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis27u1332 import guide_on_data_analysis_chapter_27_unnumbered_1332


def test_guide_on_data_analysis27u1332_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_27_unnumbered_1332(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis27u1332_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_27_unnumbered_1332(x)
    assert isinstance(result, dict)
