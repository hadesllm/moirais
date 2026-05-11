"""Tests for grnsp.geron_bert_nsp_loss."""
import numpy as np
import pytest
from morie.fn.grnsp import geron_bert_nsp_loss


def test_grnsp_basic():
    """Test basic functionality."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    labels = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_bert_nsp_loss(logits, labels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grnsp_edge():
    """Test edge cases."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    labels = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_bert_nsp_loss(logits, labels)
    assert isinstance(result, dict)
