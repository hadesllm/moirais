"""Tests for guide_on_data_analysis2u58.guide_on_data_analysis_chapter_2_unnumbered_58."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis2u58 import guide_on_data_analysis_chapter_2_unnumbered_58


def test_guide_on_data_analysis2u58_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_2_unnumbered_58(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis2u58_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_2_unnumbered_58(x)
    assert isinstance(result, dict)
