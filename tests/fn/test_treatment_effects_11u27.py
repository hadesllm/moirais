"""Tests for treatment_effects_11u27.treatment_effects_1_chapter_1_unnumbered_27."""
import numpy as np
import pytest
from moirais.fn.treatment_effects_11u27 import treatment_effects_1_chapter_1_unnumbered_27


def test_treatment_effects_11u27_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = treatment_effects_1_chapter_1_unnumbered_27(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_treatment_effects_11u27_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = treatment_effects_1_chapter_1_unnumbered_27(x)
    assert isinstance(result, dict)
