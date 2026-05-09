"""Tests for moirais.fn.ampcl — amplitude classification."""
import numpy as np
import pytest

from moirais.fn.ampcl import amplitude_classify, ampcl


def test_default_terciles():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(300)
    result = amplitude_classify(x)
    assert result.extra["n_classes"] == 3
    assert len(result.extra["labels"]) == 300


def test_custom_thresholds():
    x = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
    result = amplitude_classify(x, thresholds=[0.5, 1.5])
    assert result.extra["n_classes"] == 3


def test_all_same():
    x = np.zeros(50)
    result = amplitude_classify(x)
    assert np.all(result.extra["labels"] == 0)


def test_alias():
    assert ampcl is amplitude_classify
