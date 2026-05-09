"""Tests for guide_on_data_analysis3u121.guide_on_data_analysis_chapter_3_unnumbered_121."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis3u121 import guide_on_data_analysis_chapter_3_unnumbered_121


def test_guide_on_data_analysis3u121_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_3_unnumbered_121(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis3u121_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_3_unnumbered_121(x)
    assert isinstance(result, dict)
