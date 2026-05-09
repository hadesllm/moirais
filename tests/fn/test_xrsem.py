"""Tests for moirais.fn.xrsem -- SEM (Spatial Error) model ML estimation"""

import numpy as np
import pytest

from moirais.fn.xrsem import sem_ml


class TestSemMl:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sem_ml(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = sem_ml(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
