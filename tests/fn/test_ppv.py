"""Tests for morie.fn.ppv — Positive and negative predictive values."""
import numpy as np

from morie.fn.ppv import ppv_npv, ppv


def test_known_ppv_npv():
    """TP=8, FP=2, TN=7, FN=3 -> PPV=0.8, NPV=0.7."""
    y_true = [1] * 8 + [1] * 3 + [0] * 2 + [0] * 7
    y_pred = [1] * 8 + [0] * 3 + [1] * 2 + [0] * 7
    result = ppv_npv(y_true, y_pred)
    assert abs(result.extra["ppv"] - 0.8) < 1e-10
    assert abs(result.extra["npv"] - 0.7) < 1e-10
    assert result.extra["tp"] == 8
    assert result.extra["fp"] == 2


def test_ppv_alias():
    assert ppv is ppv_npv
