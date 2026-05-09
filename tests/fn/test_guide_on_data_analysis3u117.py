"""Tests for guide_on_data_analysis3u117.guide_on_data_analysis_chapter_3_unnumbered_117."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis3u117 import guide_on_data_analysis_chapter_3_unnumbered_117


def test_guide_on_data_analysis3u117_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_3_unnumbered_117(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_guide_on_data_analysis3u117_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_3_unnumbered_117(x)
    assert isinstance(result, dict)
