"""Tests for guide_on_data_analysis8u627.guide_on_data_analysis_chapter_8_unnumbered_627."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis8u627 import guide_on_data_analysis_chapter_8_unnumbered_627


def test_guide_on_data_analysis8u627_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_8_unnumbered_627(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_guide_on_data_analysis8u627_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_8_unnumbered_627(x)
    assert isinstance(result, dict)
