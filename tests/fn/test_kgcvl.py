"""Tests for moirais.fn.kgcvl -- Kriging LOO cross-validation"""

import numpy as np
import pytest

from moirais.fn.kgcvl import kriging_cv_loo


class TestKrigingCvLoo:
    def test_basic(self):
        vals = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = kriging_cv_loo(vals, x)
        assert result.statistic is not None

    def test_output_type(self):
        result = kriging_cv_loo(np.array([1.,2.,3.]), np.array([0.,1.,2.]))
        assert hasattr(result, "statistic")
