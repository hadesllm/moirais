"""Tests for moirais.fn.fst -- Fst fixation index."""

import numpy as np
import pytest
from moirais.fn.fst import fixation_index


class TestFst:
    def test_identical_populations(self):
        """Two identical populations should have Fst = 0."""
        freqs = np.array([[0.5, 0.3], [0.5, 0.3]])
        res = fixation_index(freqs)
        assert res.name == "Fst"
        assert res.statistic == pytest.approx(0.0, abs=1e-10)

    def test_divergent_populations(self):
        """Populations with very different allele freqs => high Fst."""
        freqs = np.array([[0.99, 0.01], [0.01, 0.99]])
        res = fixation_index(freqs)
        assert res.statistic > 0.5

    def test_single_locus(self):
        """Works with a 1D array (single locus)."""
        freqs = [0.2, 0.8]
        res = fixation_index(freqs)
        assert res.statistic > 0.0
        assert res.extra["n_loci"] == 1

    def test_too_few_pops(self):
        with pytest.raises(ValueError):
            fixation_index(np.array([[0.5, 0.3]]))
