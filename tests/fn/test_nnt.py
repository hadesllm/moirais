"""Tests for moirais.fn.nnt -- number needed to treat."""

import pytest
from moirais.fn.nnt import number_needed_to_treat


class TestNNT:
    def test_known_values(self):
        """a=20,b=80,c=10,d=90: CER=0.1, EER=0.2, RD=0.1, NNT=10."""
        result = number_needed_to_treat(a=20, b=80, c=10, d=90)
        assert result.measure == "NNT"
        assert result.estimate == pytest.approx(10.0, rel=0.01)

    def test_returns_esres(self):
        """Should return ESRes with CI."""
        result = number_needed_to_treat(a=30, b=70, c=15, d=85)
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        assert result.ci_lower < result.estimate

    def test_equal_proportions_inf(self):
        """Equal proportions give RD=0, NNT=inf."""
        import numpy as np
        result = number_needed_to_treat(a=20, b=80, c=20, d=80)
        assert result.estimate == np.inf or result.estimate > 1e6
