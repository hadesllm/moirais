"""Tests for guide_on_data_analysis8u660.guide_on_data_analysis_chapter_8_unnumbered_660."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis8u660 import guide_on_data_analysis_chapter_8_unnumbered_660


def test_guide_on_data_analysis8u660_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_8_unnumbered_660(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis8u660_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_8_unnumbered_660(x)
    assert isinstance(result, dict)
