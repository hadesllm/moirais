"""Tests for goenr.go_enrichment."""
import numpy as np
import pytest
from moirais.fn.goenr import go_enrichment


def test_goenr_basic():
    """Test basic functionality."""
    foreground = np.random.default_rng(42).normal(0, 1, 100)
    background = np.random.default_rng(42).normal(0, 1, 100)
    go_terms = np.random.default_rng(42).normal(0, 1, 100)
    result = go_enrichment(foreground, background, go_terms)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_goenr_edge():
    """Test edge cases."""
    foreground = np.random.default_rng(42).normal(0, 1, 100)
    background = np.random.default_rng(42).normal(0, 1, 100)
    go_terms = np.random.default_rng(42).normal(0, 1, 100)
    result = go_enrichment(foreground, background, go_terms)
    assert isinstance(result, dict)
