"""Tests for qrntcq.quarantine_efficacy."""
import numpy as np
import pytest
from moirais.fn.qrntcq import quarantine_efficacy


def test_qrntcq_basic():
    """Test basic functionality."""
    incubation = np.random.default_rng(42).normal(0, 1, 100)
    quarantine_duration = np.random.default_rng(42).normal(0, 1, 100)
    result = quarantine_efficacy(incubation, quarantine_duration)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_qrntcq_edge():
    """Test edge cases."""
    incubation = np.random.default_rng(42).normal(0, 1, 100)
    quarantine_duration = np.random.default_rng(42).normal(0, 1, 100)
    result = quarantine_efficacy(incubation, quarantine_duration)
    assert isinstance(result, dict)
