"""Tests for moirais.fn.mcfmt — multi-class confusion matrix."""
import numpy as np
import pytest
from moirais.fn.mcfmt import multiclass_confusion_matrix, mcfmt


def test_perfect_3class():
    r = multiclass_confusion_matrix([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2])
    cm = r.value
    assert cm == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_labels_auto_detected():
    r = multiclass_confusion_matrix([0, 1, 2], [0, 1, 2])
    assert r.extra["labels"] == [0, 1, 2]


def test_explicit_labels():
    r = multiclass_confusion_matrix([1, 2], [1, 2], labels=[1, 2, 3])
    assert len(r.value) == 3


def test_per_class_counts():
    r = multiclass_confusion_matrix([0, 0, 1, 1], [0, 1, 0, 1])
    pc = r.extra["per_class"]
    assert pc["0"]["tp"] == 1
    assert pc["0"]["fp"] == 1


def test_alias():
    assert mcfmt is multiclass_confusion_matrix


def test_length_mismatch():
    with pytest.raises(ValueError):
        multiclass_confusion_matrix([1], [0, 1])
