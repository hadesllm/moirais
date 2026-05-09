"""Tests for kminst.kamath_instruction_tuning_loss."""
import numpy as np
import pytest
from moirais.fn.kminst import kamath_instruction_tuning_loss


def test_kminst_basic():
    """Test basic functionality."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    response_mask = np.random.default_rng(42).normal(0, 1, 100)
    targets = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_instruction_tuning_loss(logits, response_mask, targets)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kminst_edge():
    """Test edge cases."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    response_mask = np.random.default_rng(42).normal(0, 1, 100)
    targets = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_instruction_tuning_loss(logits, response_mask, targets)
    assert isinstance(result, dict)
