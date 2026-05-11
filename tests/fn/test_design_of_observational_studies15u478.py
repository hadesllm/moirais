"""Tests for design_of_observational_studies15u478.design_of_observational_studies_chapter_15_unnumbered_478."""
import numpy as np
import pytest
from morie.fn.design_of_observational_studies15u478 import design_of_observational_studies_chapter_15_unnumbered_478


def test_design_of_observational_studies15u478_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_478(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies15u478_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_478(x)
    assert isinstance(result, dict)
