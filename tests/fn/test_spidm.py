"""Tests for morie.fn.spidm -- web graph community detection."""

import numpy as np
from morie.fn.spidm import web_graph_communities, spidm
from morie.fn._containers import DescriptiveResult


class TestSpidm:
    def test_alias(self):
        assert spidm is web_graph_communities

    def test_two_communities(self):
        A = np.zeros((6, 6))
        for i in range(3):
            for j in range(3):
                if i != j:
                    A[i, j] = 1
        for i in range(3, 6):
            for j in range(3, 6):
                if i != j:
                    A[i, j] = 1
        A[2, 3] = A[3, 2] = 0.1
        r = web_graph_communities(A, k=2, seed=42)
        assert isinstance(r, DescriptiveResult)
        labels = r.value["labels"]
        assert len(labels) == 6
        assert len(np.unique(labels)) == 2

    def test_modularity(self):
        A = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]])
        r = web_graph_communities(A, k=2, seed=0)
        assert "modularity" in r.value
