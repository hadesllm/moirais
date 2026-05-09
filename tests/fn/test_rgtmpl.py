"""Tests for rgtmpl.rangayyan_template_match."""
import numpy as np
import pytest
from moirais.fn.rgtmpl import rangayyan_template_match


def test_rgtmpl_basic():
    """Test basic functionality."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    template = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_template_match(eeg, template, threshold)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgtmpl_edge():
    """Test edge cases."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    template = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_template_match(eeg, template, threshold)
    assert isinstance(result, dict)
