"""Tests for precision_recall."""
import pytest
from morie.fn.prrcl import precision_recall, prrcl


def test_perfect():
    r = precision_recall([1, 0, 1, 0], [1, 0, 1, 0])
    assert r.extra["precision"] == 1.0
    assert r.extra["recall"] == 1.0
    assert r.estimate == 1.0


def test_alias():
    assert prrcl is precision_recall


def test_no_positives():
    r = precision_recall([0, 0], [0, 0])
    assert r.extra["precision"] == 0.0
