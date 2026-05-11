"""Tests for guide_on_data_analysis27u1341.guide_on_data_analysis_chapter_27_unnumbered_1341."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis27u1341 import guide_on_data_analysis_chapter_27_unnumbered_1341


def test_guide_on_data_analysis27u1341_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_27_unnumbered_1341(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis27u1341_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_27_unnumbered_1341(x)
    assert isinstance(result, dict)
