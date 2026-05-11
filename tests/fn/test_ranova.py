"""Tests for ranova.random_effects_anova_decomp."""
import numpy as np
import pytest
from morie.fn.ranova import random_effects_anova_decomp


def test_ranova_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = random_effects_anova_decomp(y, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ranova_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = random_effects_anova_decomp(y, cluster)
    assert isinstance(result, dict)
