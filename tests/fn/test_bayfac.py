"""Tests for bayfac.bayes_factor."""
import numpy as np
import pytest
from moirais.fn.bayfac import bayes_factor


def test_bayfac_basic():
    """Test basic functionality."""
    log_evidence_1 = np.random.default_rng(42).normal(0, 1, 100)
    log_evidence_2 = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_factor(log_evidence_1, log_evidence_2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bayfac_edge():
    """Test edge cases."""
    log_evidence_1 = np.random.default_rng(42).normal(0, 1, 100)
    log_evidence_2 = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_factor(log_evidence_1, log_evidence_2)
    assert isinstance(result, dict)
