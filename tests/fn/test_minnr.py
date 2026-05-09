"""Tests for min_distance_classify."""
import numpy as np
from moirais.fn.minnr import min_distance_classify, minnr


def test_basic():
    X_train = [[0, 0], [1, 1], [10, 10], [11, 11]]
    y_train = [0, 0, 1, 1]
    X_test = [[0.5, 0.5], [10.5, 10.5]]
    r = min_distance_classify(X_train, y_train, X_test)
    preds = r.extra["predictions"]
    assert preds[0] == 0
    assert preds[1] == 1


def test_alias():
    assert minnr is min_distance_classify
