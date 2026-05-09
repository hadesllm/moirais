"""He who has a why to live can bear almost any how. — Friedrich Nietzsche"""

import numpy as np
import pytest
from moirais.fn.thorm import bonferroni_correction, thorm
from moirais.fn._containers import DescriptiveResult


class TestThorm:
    def test_alias(self):
        assert thorm is bonferroni_correction

    def test_basic_correction(self):
        p = [0.01, 0.04, 0.03]
        r = bonferroni_correction(p)
        assert isinstance(r, DescriptiveResult)
        adj = r.value["adjusted"]
        assert np.allclose(adj, [0.03, 0.12, 0.09])
        assert r.value["n_rejected"] == 1

    def test_invalid_pvalues(self):
        with pytest.raises(ValueError):
            bonferroni_correction([0.5, 1.5])
