"""Tests for morie.fn.vface — Content validity index."""

import numpy as np
import pytest
from morie.fn.vface import validity_face_content


class TestValidityFaceContent:

    def test_perfect_ratings(self):
        items = ["a", "b", "c"]
        ratings = np.array([[4, 4, 4], [3, 4, 3], [4, 3, 4]])
        result = validity_face_content(items, ratings)
        assert result["s_cvi_ave"] == 1.0

    def test_i_cvi_per_item(self):
        items = ["a", "b"]
        ratings = np.array([[4, 1], [3, 2], [4, 1], [3, 2]])
        result = validity_face_content(items, ratings)
        assert result["i_cvi"]["a"] == 1.0
        assert result["i_cvi"]["b"] == 0.0

    def test_acceptable_items_filtered(self):
        items = ["good", "bad"]
        ratings = np.array([[4, 1], [3, 2], [4, 1]])
        result = validity_face_content(items, ratings, threshold=0.78)
        assert "good" in result["acceptable_items"]
        assert "bad" not in result["acceptable_items"]

    def test_n_experts_correct(self):
        items = ["a"]
        ratings = np.array([[4], [3], [4], [3], [4]])
        result = validity_face_content(items, ratings)
        assert result["n_experts"] == 5

    def test_items_length_mismatch(self):
        with pytest.raises(ValueError):
            validity_face_content(["a", "b"], np.array([[4, 3, 2]]))
