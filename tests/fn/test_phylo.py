"""Tests for morie.fn.phylo -- phylogenetic diversity."""

import numpy as np
import pytest
from morie.fn.phylo import phylogenetic_diversity


class TestPhyloDiversity:
    def test_two_taxa(self):
        D = np.array([[0.0, 3.0], [3.0, 0.0]])
        res = phylogenetic_diversity(D)
        assert res["pd"] == pytest.approx(3.0)
        assert res["n_taxa"] == 2

    def test_three_equidistant(self):
        D = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        res = phylogenetic_diversity(D)
        assert res["pd"] == pytest.approx(2.0)

    def test_mst_edges(self):
        D = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]], dtype=float)
        res = phylogenetic_diversity(D)
        assert len(res["mst_edges"]) == 2
        assert res["pd"] == pytest.approx(3.0)

    def test_mean_distance(self):
        D = np.array([[0, 2, 4], [2, 0, 6], [4, 6, 0]], dtype=float)
        res = phylogenetic_diversity(D)
        assert res["mean_distance"] == pytest.approx(4.0)

    def test_single_taxon_raises(self):
        with pytest.raises(ValueError):
            phylogenetic_diversity(np.array([[0.0]]))

    def test_nonsquare_raises(self):
        with pytest.raises(ValueError):
            phylogenetic_diversity(np.array([[0, 1], [1, 0], [2, 2]]))
