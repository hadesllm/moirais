"""Tests for difrj -- Raju's area DIF."""
import numpy as np
from morie.fn.difrj import dif_raju_area
from morie.fn._containers import DIFResult


class TestDifRaju:
    def test_no_dif(self):
        a = np.ones(5)
        b = np.zeros(5)
        result = dif_raju_area(a, b, a, b)
        assert isinstance(result, DIFResult)
        assert result.method == "Raju"

    def test_signed_area(self):
        result = dif_raju_area([1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0])
        df = result.items
        assert "signed_area" in df.columns
        assert len(df) == 2
