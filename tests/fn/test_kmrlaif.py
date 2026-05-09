"""Tests for kmrlaif.kamath_rlaif_objective."""
import numpy as np
import pytest
from moirais.fn.kmrlaif import kamath_rlaif_objective


def test_kmrlaif_basic():
    """Test basic functionality."""
    ai_preferences = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_rlaif_objective(ai_preferences)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmrlaif_edge():
    """Test edge cases."""
    ai_preferences = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_rlaif_objective(ai_preferences)
    assert isinstance(result, dict)
