"""Tests for design_of_observational_studies6u398.design_of_observational_studies_chapter_6_unnumbered_398."""
import numpy as np
import pytest
from morie.fn.design_of_observational_studies6u398 import design_of_observational_studies_chapter_6_unnumbered_398


def test_design_of_observational_studies6u398_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_6_unnumbered_398(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies6u398_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_6_unnumbered_398(x)
    assert isinstance(result, dict)
