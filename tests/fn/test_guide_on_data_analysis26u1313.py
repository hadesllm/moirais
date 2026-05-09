"""Tests for guide_on_data_analysis26u1313.guide_on_data_analysis_chapter_26_unnumbered_1313."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis26u1313 import guide_on_data_analysis_chapter_26_unnumbered_1313


def test_guide_on_data_analysis26u1313_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_26_unnumbered_1313(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis26u1313_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_26_unnumbered_1313(x)
    assert isinstance(result, dict)
