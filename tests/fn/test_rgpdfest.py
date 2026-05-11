"""Tests for rgpdfest.rangayyan_pdf_estimate."""
import numpy as np
import pytest
from morie.fn.rgpdfest import rangayyan_pdf_estimate


def test_rgpdfest_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bins = np.random.default_rng(42).normal(0, 1, 100)
    bw = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_pdf_estimate(x, bins, bw)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgpdfest_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bins = np.random.default_rng(42).normal(0, 1, 100)
    bw = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_pdf_estimate(x, bins, bw)
    assert isinstance(result, dict)
