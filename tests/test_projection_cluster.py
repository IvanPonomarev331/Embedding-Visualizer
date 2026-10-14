"""Numerical behavior, geometry, and invalid public-operation parameters."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from embedding_visualizer.compute.clustering import cluster
from embedding_visualizer.compute.common import ComputeError
from embedding_visualizer.compute.projections import project


@pytest.fixture
def blobs():
    return make_blobs(n_samples=90, n_features=5, centers=3, cluster_std=0.18, random_state=7)


def test_pca_retains_geometry_and_row_order():
    matrix = np.array([[1, 2, 1], [2, 4, 2], [3, 6, 3], [4, 8, 4]], dtype=float)
    before = matrix.copy()
    result = project(matrix, "pca", n_components=2)
    assert result.coordinates.shape == (4, 2)
    assert result.explained_variance_ratio[0] == pytest.approx(1)
    original_distances = np.linalg.norm(matrix[:, None] - matrix[None, :], axis=-1)
    projected_distances = np.linalg.norm(result.coordinates[:, None] - result.coordinates[None, :], axis=-1)
    np.testing.assert_allclose(projected_distances, original_distances, atol=1e-12)
    np.testing.assert_array_equal(matrix, before)
    np.testing.assert_array_equal(project(matrix, "pca").coordinates, result.coordinates)


def test_constant_pca_is_finite_and_explains_no_variance():
    result = project(np.ones((4, 3)), "pca")
    assert np.all(result.coordinates == 0)
    assert np.all(result.explained_variance_ratio == 0)
    assert "no variance" in result.warnings[0]


@pytest.mark.parametrize("method,params", [
    ("umap", {"n_neighbors": 2}),
    ("tsne", {"perplexity": 2.0, "max_iter": 300}),
])
def test_nonlinear_small_dataset_finite_and_reproducible(method, params):
    matrix = np.random.default_rng(3).normal(size=(8, 4))
    before = matrix.copy()
    first = project(matrix, method, random_state=2, **params)
    second = project(matrix, method, random_state=2, **params)
    assert first.coordinates.shape == (8, 2)
    assert np.isfinite(first.coordinates).all()
    np.testing.assert_allclose(first.coordinates, second.coordinates, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(matrix, before)
    assert first.explained_variance_ratio is None
    assert first.parameters["metric"] == "euclidean"


def test_umap_minimal_dataset_avoids_spectral_initialization_failure():
    result = project([[1, 1], [2, 4], [3, -1]], "umap")
    assert result.coordinates.shape == (3, 2)
    assert np.isfinite(result.coordinates).all()
    assert result.parameters["init"] == "random"


def test_tsne_two_rows_and_more_components_than_features():
    result = project([[1.0], [2.0]], "tsne", max_iter=300, n_components=4)
    assert result.coordinates.shape == (2, 4)
    assert np.isfinite(result.coordinates).all()
    assert result.parameters["method"] == "exact"


@pytest.mark.parametrize("method,params,error", [
    ("unknown", {}, "Projection method"),
    ("pca", {"n_components": 5}, "n_components"),
    ("pca", {"metric": "cosine"}, "Unsupported"),
    ("pca", {"random_state": True}, "random_state"),
    ("pca", {"n_components": 1.2}, "n_components"),
    ("umap", {"n_neighbors": 4}, "n_neighbors"),
    ("umap", {"n_neighbors": 1}, "n_neighbors"),
    ("umap", {"min_dist": 1.1}, "min_dist"),
    ("tsne", {"perplexity": 4.0}, "perplexity"),
    ("tsne", {"perplexity": 0.0}, "perplexity"),
    ("tsne", {"max_iter": 249}, "max_iter"),
    ("tsne", {"random_state": 2**32}, "random_state"),
    ("tsne", {"metric": "dot_product"}, "metric"),
])
def test_invalid_projection_parameters(method, params, error):
    with pytest.raises(ComputeError, match=error):
        project(np.arange(12).reshape(4, 3), method, **params)


@pytest.mark.parametrize("method", ["umap", "tsne"])
def test_cosine_projection_rejects_zero_vectors(method):
    with pytest.raises(ComputeError, match="zero vectors"):
        project([[0, 0], [1, 1], [2, 1]], method, metric="cosine")


def test_projection_rejects_invalid_input():
    with pytest.raises(ComputeError):
        project([[np.nan, 1], [1, 2]], "pca")
    with pytest.raises(ComputeError, match="at least 2"):
        project([[1, 2]], "pca")
    with pytest.raises(ComputeError, match="at least 3"):
        project([[1, 2], [3, 4]], "umap")


def test_kmeans_known_blobs_and_reproducibility(blobs):
    matrix, truth = blobs
    before = matrix.copy()
    first = cluster(matrix, "kmeans", n_clusters=3)
    second = cluster(matrix, "kmeans", n_clusters=3)
    assert adjusted_rand_score(truth, first.labels) == 1
    np.testing.assert_array_equal(first.labels, second.labels)
    np.testing.assert_array_equal(matrix, before)
    assert first.parameters["metric"] == "euclidean"
    assert first.space == "original"
    np.testing.assert_array_equal(first.representation, matrix)


def test_clustering_explicit_pca_matches_clustering_that_space(blobs):
    matrix, truth = blobs
    pca_result = project(matrix, "pca", n_components=2)
    expected = cluster(pca_result.coordinates, "kmeans", n_clusters=3)
    result = cluster(matrix, "kmeans", space="pca", pca_components=2, n_clusters=3)
    assert adjusted_rand_score(truth, result.labels) == 1
    np.testing.assert_array_equal(result.labels, expected.labels)
    assert result.space == "pca"
    assert result.parameters["pca_components"] == 2
    np.testing.assert_allclose(result.representation, pca_result.coordinates)


def test_hdbscan_detects_dense_groups_and_noise():
    rng = np.random.default_rng(42)
    matrix = np.concatenate([
        rng.normal([-5, 0], 0.05, size=(30, 2)),
        rng.normal([5, 0], 0.05, size=(30, 2)),
        [[0, 100]],
    ])
    before = matrix.copy()
    result = cluster(matrix, "hdbscan", min_cluster_size=8, min_samples=5)
    assert result.labels[-1] == -1
    assert adjusted_rand_score([0] * 30 + [1] * 30, result.labels[:-1]) == 1
    np.testing.assert_array_equal(matrix, before)


def test_hdbscan_cosine_supported_with_brute():
    rng = np.random.default_rng(10)
    matrix = np.concatenate([rng.normal([3, 0], .03, (20, 2)), rng.normal([0, 3], .03, (20, 2))])
    result = cluster(matrix, "hdbscan", metric="cosine", min_cluster_size=5, min_samples=4)
    assert adjusted_rand_score([0] * 20 + [1] * 20, result.labels) == 1
    assert result.parameters["algorithm"] == "brute"


def test_hdbscan_all_noise_returns_actionable_warning():
    result = cluster([[1, 0], [0, 1]], "hdbscan", min_cluster_size=2)
    assert list(result.labels) == [-1, -1]
    assert "every object to noise" in result.warnings[-1]


@pytest.mark.parametrize("method,params,error", [
    ("other", {}, "Clustering method"),
    ("kmeans", {"n_clusters": 0}, "n_clusters"),
    ("kmeans", {"n_clusters": 5}, "n_clusters"),
    ("kmeans", {"metric": "cosine"}, "Unsupported"),
    ("kmeans", {"space": "umap"}, "Clustering space"),
    ("kmeans", {"space": "pca"}, "pca_components"),
    ("kmeans", {"pca_components": 2}, "only valid"),
    ("kmeans", {"space": "pca", "pca_components": 8}, "n_components"),
    ("hdbscan", {"min_cluster_size": 1}, "min_cluster_size"),
    ("hdbscan", {"min_samples": 5}, "min_samples"),
    ("hdbscan", {"min_samples": 0}, "min_samples"),
    ("hdbscan", {"metric": "dot_product"}, "metric"),
    ("hdbscan", {"metric": "cosine"}, "zero vectors"),
])
def test_invalid_clustering_parameters(method, params, error):
    with pytest.raises(ComputeError, match=error):
        cluster([[0, 0], [1, 1], [2, 3], [4, 1]], method, **params)


def test_hdbscan_cosine_memory_budget_rejects_before_estimator_fit(monkeypatch):
    from embedding_visualizer.compute import clustering

    calls = []

    def fit_predict(estimator, matrix):
        calls.append(estimator.metric)
        return np.zeros(len(matrix), dtype=np.int64)

    monkeypatch.setattr(clustering.HDBSCAN, "fit_predict", fit_predict)
    matrix = np.ones((8193, 2))
    with pytest.raises(ComputeError, match="512 MiB.*Choose Euclidean"):
        cluster(matrix, "hdbscan", metric="cosine")
    assert calls == []
    cluster(matrix, "hdbscan", metric="euclidean")
    assert calls == ["euclidean"]
