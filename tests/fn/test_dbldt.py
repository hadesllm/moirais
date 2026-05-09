"""Tests for moirais.fn.dbldt — double-threshold detection."""
import numpy as np
import pytest

from moirais.fn.dbldt import double_threshold, dbldt


def test_single_event():
    x = np.array([0, 0, 0.8, 0.9, 0.5, 0.2, 0, 0])
    result = double_threshold(x, low=0.3, high=0.7)
    assert result.extra["n_events"] == 1


def test_no_events():
    x = np.zeros(50)
    result = double_threshold(x, low=0.3, high=0.7)
    assert result.extra["n_events"] == 0


def test_mask_shape():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 100)
    result = double_threshold(x, low=0.3, high=0.7)
    assert len(result.extra["mask"]) == 100


def test_alias():
    assert dbldt is double_threshold
