"""Known geometry and invalid input at the independent compute boundary."""

import numpy as np
import pytest

from embedding_visualizer.compute.common import ComputeError, prepare_embeddings, validate_matrix
from embedding_visualizer.compute.neighbors import exact_neighbors


@pytest.mark.parametrize("bad", [
    [], [[]], [1, 2], [[[1]]], [[1, 2], [1]], [[1, np.nan]], [[np.inf]],
    [[True]], [[1, True]], [["1"]], [[1 + 0j]], np.array([[1]], dtype=object),
])
def test_rejects_invalid_matrices(bad):
    with pytest.raises(ComputeError):
        validate_matrix(bad)


def test_normalization_is_explicit_stable_and_nonmutating():
    matrix = np.array([[3, 4], [0, 0], [3e38, 4e38], [3e-38, 4e-38]], dtype=np.float64)
    before = matrix.copy()
    normalized = prepare_embeddings(matrix, normalize=True)
    np.testing.assert_allclose(normalized, [[0.6, 0.8], [0, 0], [0.6, 0.8], [0.6, 0.8]])
    np.testing.assert_array_equal(matrix, before)
    unnormalized = prepare_embeddings(matrix)
    unnormalized[0, 0] = 99
    np.testing.assert_array_equal(matrix, before)
    np.testing.assert_allclose(prepare_embeddings([[1e308, 1e308]], True), [[2 ** -0.5, 2 ** -0.5]])
    np.testing.assert_allclose(prepare_embeddings([[1e-308, 1e-308]], True), [[2 ** -0.5, 2 ** -0.5]])
    with pytest.raises(ComputeError, match="boolean"):
        prepare_embeddings(matrix, normalize="false")


def test_cosine_known_values_excludes_self_and_zero_with_deterministic_ties():
    matrix = [[1, 0], [0, 1], [2, 0], [-1, 0], [1, 1], [0, 0], [3, 0]]
    result = exact_neighbors(matrix, 0, k=20)
    np.testing.assert_array_equal(result.indices, [2, 6, 4, 1, 3])
    np.testing.assert_allclose(result.values, [1, 1, 2 ** -0.5, 0, -1])
    assert result.metric == "cosine"
    assert result.value_kind == "similarity"
    assert result.excluded_zero_count == 1
    with pytest.raises(ComputeError, match="zero query"):
        exact_neighbors(matrix, 5)


def test_euclidean_known_values_includes_zero_and_ties_by_row():
    matrix = [[0, 0], [3, 4], [0, 5], [1, 0], [0, 0]]
    result = exact_neighbors(matrix, 0, k=4, metric="euclidean")
    np.testing.assert_array_equal(result.indices, [4, 3, 1, 2])
    np.testing.assert_allclose(result.values, [0, 1, 5, 5])
    assert result.value_kind == "distance"
    assert result.excluded_zero_count == 0


def test_k_cutoff_ties_select_earliest_original_rows():
    result = exact_neighbors([[1, 0], [2, 0], [3, 0], [4, 0]], 3, k=2)
    np.testing.assert_array_equal(result.indices, [0, 1])


def test_dot_product_differs_from_cosine_and_uses_float64():
    matrix = np.array([[1, 0], [2, 0], [3, 4], [0, 0], [-1, 0]], dtype=np.float32)
    result = exact_neighbors(matrix, 0, k=3, metric="dot_product")
    np.testing.assert_array_equal(result.indices, [2, 1, 3])
    np.testing.assert_array_equal(result.values, [3, 2, 0])
    assert result.value_kind == "similarity"
    large = exact_neighbors(np.array([[1e30], [2e30]], dtype=np.float32), 0, metric="dot_product")
    assert large.values.dtype == np.float64
    assert np.isfinite(large.values).all()


@pytest.mark.parametrize("metric", ["cosine", "euclidean", "dot_product"])
def test_single_row_returns_empty_result(metric):
    result = exact_neighbors([[1, 2]], 0, metric=metric)
    assert result.indices.shape == result.values.shape == (0,)


@pytest.mark.parametrize("arguments", [
    {"query_index": -1}, {"query_index": 2}, {"query_index": True},
    {"query_index": 0.0}, {"query_index": 0, "k": 0},
    {"query_index": 0, "k": True}, {"query_index": 0, "k": 1.2},
    {"query_index": 0, "metric": "l2"}, {"query_index": 0, "metric": []},
])
def test_rejects_bad_search_arguments(arguments):
    with pytest.raises(ComputeError):
        exact_neighbors([[1, 0], [0, 1]], **arguments)


def test_numeric_overflow_has_actionable_error():
    with pytest.raises(ComputeError, match="normalization"):
        exact_neighbors([[1e308], [1e308]], 0, metric="dot_product")
