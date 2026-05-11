"""Tests for knn_classify."""
import numpy as np
from morie.fn.knnc import knn_classify, knnc


def test_basic():
    X_train = [[0], [1], [2], [10], [11], [12]]
    y_train = [0, 0, 0, 1, 1, 1]
    X_test = [[0.5], [11.5]]
    r = knn_classify(X_train, y_train, X_test, k=3)
    preds = r.extra["predictions"]
    assert preds[0] == 0
    assert preds[1] == 1


def test_alias():
    assert knnc is knn_classify


def test_with_accuracy():
    X_train = [[0], [1], [10], [11]]
    y_train = [0, 0, 1, 1]
    r = knn_classify(X_train, y_train, [[0.5], [10.5]], k=1, y_test=[0, 1])
    assert r.extra["accuracy"] == 1.0
