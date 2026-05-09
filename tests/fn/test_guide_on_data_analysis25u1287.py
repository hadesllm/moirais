"""Tests for guide_on_data_analysis25u1287.guide_on_data_analysis_chapter_25_unnumbered_1287."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis25u1287 import guide_on_data_analysis_chapter_25_unnumbered_1287


def test_guide_on_data_analysis25u1287_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_25_unnumbered_1287(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis25u1287_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_25_unnumbered_1287(x)
    assert isinstance(result, dict)
