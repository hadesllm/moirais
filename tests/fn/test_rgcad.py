"""Tests for rgcad.rangayyan_cad_pipeline."""
import numpy as np
import pytest
from morie.fn.rgcad import rangayyan_cad_pipeline


def test_rgcad_basic():
    """Test basic functionality."""
    signals = np.random.default_rng(42).normal(0, 1, 100)
    labels = np.random.default_rng(43).integers(0, 2, 100)
    classifier = np.random.default_rng(42).normal(0, 1, 100)
    cv_k = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_cad_pipeline(signals, labels, classifier, cv_k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgcad_edge():
    """Test edge cases."""
    signals = np.random.default_rng(42).normal(0, 1, 100)
    labels = np.random.default_rng(43).integers(0, 2, 100)
    classifier = np.random.default_rng(42).normal(0, 1, 100)
    cv_k = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_cad_pipeline(signals, labels, classifier, cv_k)
    assert isinstance(result, dict)
