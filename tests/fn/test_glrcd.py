"""Tests for glrcd.py - GLR change point detection."""
import numpy as np
import pytest
from moirais.fn.glrcd import glr_change, glrcd


def test_glr_change_returns_descriptive_result():
    rng = np.random.default_rng(42)
    x = np.concatenate([rng.standard_normal(100), rng.standard_normal(100) + 3.0])
    result = glr_change(x)
    assert result.name == "glr_change_detect"
    assert isinstance(result.value, int)
    assert "glr" in result.extra


def test_glr_change_detects_shift():
    rng = np.random.default_rng(42)
    x = np.concatenate([rng.standard_normal(100), rng.standard_normal(100) + 5.0])
    result = glr_change(x, min_segment=20)
    assert 50 <= result.value <= 150


def test_glr_change_glr_positive():
    rng = np.random.default_rng(42)
    x = np.concatenate([rng.standard_normal(100), rng.standard_normal(100) + 3.0])
    result = glr_change(x)
    assert result.extra["glr"] >= 0


def test_glrcd_alias():
    rng = np.random.default_rng(42)
    x = np.concatenate([rng.standard_normal(80), rng.standard_normal(80) + 2.0])
    result = glrcd(x)
    assert result.name == "glr_change_detect"
