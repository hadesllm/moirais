"""Tests for durbw.py - Durbin-Watson statistic."""
import numpy as np
from morie.fn.durbw import durbin_watson_stat_fn, durbw


def test_durbw_returns_result():
    residuals = np.random.default_rng(42).standard_normal(256)
    result = durbin_watson_stat_fn(residuals)
    assert result.name == "durbin_watson"
    assert "dw" in result.extra


def test_durbw_white_noise_near_two():
    residuals = np.random.default_rng(42).standard_normal(1000)
    result = durbin_watson_stat_fn(residuals)
    assert 1.5 < result.extra["dw"] < 2.5


def test_durbw_alias():
    residuals = np.random.default_rng(42).standard_normal(64)
    result = durbw(residuals)
    assert result.name == "durbin_watson"
