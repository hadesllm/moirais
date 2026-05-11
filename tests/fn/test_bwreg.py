"""Tests for morie.fn.bwreg — bandwidth-selected regression."""

import numpy as np
import pytest

from morie.fn.bwreg import bwreg


def test_basic_output():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    y = 2 * x + rng.standard_normal(100) * 0.3
    result = bwreg(x, y)
    assert "fitted" in result
    assert result["bandwidth"] > 0


def test_linear_fit():
    x = np.linspace(-2, 2, 200)
    y = 3 * x + 1
    result = bwreg(x, y, bandwidth=0.5)
    mid = len(result["eval_points"]) // 2
    assert result["fitted"][mid] == pytest.approx(1.0, abs=1.0)


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        bwreg(np.array([]), np.array([]))


def test_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        bwreg(np.array([1.0, 2.0]), np.array([1.0]))
