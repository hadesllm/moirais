"""Tests for morie.fn.seqdt — sequential change-point detection."""
import numpy as np
import pytest

from morie.fn.seqdt import sequential_detect, seqdt


def test_cusum_no_change():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    result = sequential_detect(x, threshold=10.0, method="cusum")
    assert result.extra["method"] == "cusum"


def test_cusum_detects_shift():
    rng = np.random.default_rng(42)
    x = np.concatenate([rng.standard_normal(100), rng.standard_normal(100) + 5])
    result = sequential_detect(x, threshold=3.0, method="cusum")
    assert len(result.extra["alarms"]) > 0


def test_ewma_method():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    result = sequential_detect(x, threshold=3.0, method="ewma")
    assert result.extra["method"] == "ewma"


def test_alias():
    assert seqdt is sequential_detect
