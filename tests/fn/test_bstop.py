"""Tests for bstop (BST operations)."""
import numpy as np
from moirais.fn.bstop import bst_operations


def test_bst_inorder():
    vals = np.array([5, 3, 7, 1, 4])
    r = bst_operations(vals)
    trav = r.extra["inorder_traversal"]
    assert trav == sorted(trav)


def test_bst_search():
    vals = np.array([5, 3, 7, 1, 4])
    r = bst_operations(vals, queries=np.array([3, 6]))
    assert r.extra["search_results"][3.0] is True
    assert r.extra["search_results"][6.0] is False
