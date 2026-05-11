"""Tests for morie.fn.msemb -- Embedding quality measure"""

import numpy as np
import pytest

from morie.fn.msemb import embedding_qual


class TestEmbeddingQual:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = embedding_qual(data)
        assert result.value is not None

    def test_output_type(self):
        result = embedding_qual(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
