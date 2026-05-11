"""Tests for design_of_observational_studies3u231.design_of_observational_studies_chapter_3_unnumbered_231."""
import numpy as np
import pytest
from morie.fn.design_of_observational_studies3u231 import design_of_observational_studies_chapter_3_unnumbered_231


def test_design_of_observational_studies3u231_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_3_unnumbered_231(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies3u231_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_3_unnumbered_231(x)
    assert isinstance(result, dict)
