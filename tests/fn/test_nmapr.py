"""Tests for moirais.fn.nmapr -- Aggregate PRE"""

import numpy as np
import pytest

from moirais.fn.nmapr import apre_stat


class TestApreStat:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apre_stat(data)
        assert result.value is not None

    def test_output_type(self):
        result = apre_stat(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
