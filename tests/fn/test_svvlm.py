"""Tests for morie.fn.svvlm -- Valence advantage model (Groseclose)"""

import numpy as np
import pytest

from morie.fn.svvlm import valence_model


class TestValenceModel:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = valence_model(data)
        assert result.value is not None

    def test_output_type(self):
        result = valence_model(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
