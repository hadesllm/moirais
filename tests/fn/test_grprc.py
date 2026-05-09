"""Tests for grprc.geron_precision_recall_curve."""
import numpy as np
import pytest
from moirais.fn.grprc import geron_precision_recall_curve


def test_grprc_basic():
    """Test basic functionality."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_scores = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_precision_recall_curve(y_true, y_scores)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grprc_edge():
    """Test edge cases."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_scores = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_precision_recall_curve(y_true, y_scores)
    assert isinstance(result, dict)
