"""Tests for design_of_observational_studies17u649.design_of_observational_studies_chapter_17_unnumbered_649."""
import numpy as np
import pytest
from morie.fn.design_of_observational_studies17u649 import design_of_observational_studies_chapter_17_unnumbered_649


def test_design_of_observational_studies17u649_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_17_unnumbered_649(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies17u649_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_17_unnumbered_649(x)
    assert isinstance(result, dict)
