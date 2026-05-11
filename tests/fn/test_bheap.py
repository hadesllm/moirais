"""Tests for bheap (binary heap)."""
import numpy as np
from morie.fn.bheap import binary_heap


def test_binary_heap_min():
    data = np.array([5, 3, 8, 1, 4])
    r = binary_heap(data, heap_type="min")
    sorted_out = r.extra["sorted"]
    assert sorted_out == sorted(sorted_out)


def test_binary_heap_max():
    data = np.array([5, 3, 8, 1, 4])
    r = binary_heap(data, heap_type="max")
    sorted_out = r.extra["sorted"]
    assert sorted_out == sorted(sorted_out, reverse=True)
