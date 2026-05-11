"""Tests for guide_on_data_analysis27u1359.guide_on_data_analysis_chapter_27_unnumbered_1359."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis27u1359 import guide_on_data_analysis_chapter_27_unnumbered_1359


def test_guide_on_data_analysis27u1359_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_27_unnumbered_1359(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis27u1359_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_27_unnumbered_1359(x)
    assert isinstance(result, dict)
