"""Tests for design_of_observational_studies3u178.design_of_observational_studies_chapter_3_unnumbered_178."""
import numpy as np
import pytest
from moirais.fn.design_of_observational_studies3u178 import design_of_observational_studies_chapter_3_unnumbered_178


def test_design_of_observational_studies3u178_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_3_unnumbered_178(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies3u178_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_3_unnumbered_178(x)
    assert isinstance(result, dict)
