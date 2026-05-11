"""Tests for morie.fn.kgcvk -- Kriging k-fold cross-validation"""

import numpy as np
import pytest

from morie.fn.kgcvk import kriging_cv_kfold


class TestKrigingCvKfold:
    def test_basic(self):
        vals = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = kriging_cv_kfold(vals, x)
        assert result.statistic is not None

    def test_output_type(self):
        result = kriging_cv_kfold(np.array([1.,2.,3.]), np.array([0.,1.,2.]))
        assert hasattr(result, "statistic")
