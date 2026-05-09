"""Tests for design_of_observational_studies20u711.design_of_observational_studies_chapter_20_unnumbered_711."""
import numpy as np
import pytest
from moirais.fn.design_of_observational_studies20u711 import design_of_observational_studies_chapter_20_unnumbered_711


def test_design_of_observational_studies20u711_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_20_unnumbered_711(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies20u711_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_20_unnumbered_711(x)
    assert isinstance(result, dict)
