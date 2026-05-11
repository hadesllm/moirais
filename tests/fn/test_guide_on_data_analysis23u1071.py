"""Tests for guide_on_data_analysis23u1071.guide_on_data_analysis_chapter_23_unnumbered_1071."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis23u1071 import guide_on_data_analysis_chapter_23_unnumbered_1071


def test_guide_on_data_analysis23u1071_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_23_unnumbered_1071(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis23u1071_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_23_unnumbered_1071(x)
    assert isinstance(result, dict)
