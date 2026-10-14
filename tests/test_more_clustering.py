"""Geometric behavior and resource limits of the additional clustering methods."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score

from embedding_visualizer.compute import clustering
from embedding_visualizer.compute.clustering import CLUSTER_METHODS, cluster
from embedding_visualizer.compute.common import ComputeError


@pytest.fixture
def blobs():
    return make_blobs(n_samples=90, centers=3, n_features=5, cluster_std=0.12, random_state=7)


@pytest.mark.parametrize("method,params", [
    ("spectral", {"n_clusters": 3, "n_neighbors": 10}),
    ("agglomerative", {"n_clusters": 3, "linkage": "ward"}),
    ("agglomerative", {"n_clusters": 3, "linkage": "average", "metric": "cosine"}),
    ("agglomerative", {"n_clusters": 3, "linkage": "complete"}),
    ("agglomerative", {"n_clusters": 3, "linkage": "single"}),
    ("dbscan", {"eps": 0.8, "min_samples": 5}),
    ("gaussian_mixture", {"n_clusters": 3, "covariance_type": "diag"}),
    ("gaussian_mixture", {"n_clusters": 3, "covariance_type": "spherical"}),
])
def test_separated_clusters_recovered_and_input_unchanged(blobs, method, params):
    matrix, truth = blobs
    original = matrix.copy()
    result = cluster(matrix, method, **params)
    assert adjusted_rand_score(truth, result.labels) == 1
    assert result.method == method
    assert result.space == "original"
    assert result.labels.dtype == np.int64
    np.testing.assert_array_equal(matrix, original)
    np.testing.assert_array_equal(result.representation, original)


def test_spectral_recovers_nonconvex_moons_where_centroid_clustering_cannot():
    matrix, truth = make_moons(n_samples=240, noise=0.035, random_state=7)
    spectral = cluster(matrix, "spectral", n_clusters=2, n_neighbors=10)
    kmeans = cluster(matrix, "kmeans", n_clusters=2)
    assert adjusted_rand_score(truth, spectral.labels) > 0.95
    assert adjusted_rand_score(truth, kmeans.labels) < 0.7
    assert spectral.parameters["affinity"] == "nearest_neighbors"


@pytest.mark.parametrize("method,params", [
    ("spectral", {"n_clusters": 3, "n_neighbors": 10}),
    ("gaussian_mixture", {"n_clusters": 3}),
])
def test_seed_reproduces_stochastic_labels(blobs, method, params):
    matrix, _ = blobs
    first = cluster(matrix, method, random_state=23, **params)
    second = cluster(matrix, method, random_state=23, **params)
    np.testing.assert_array_equal(first.labels, second.labels)
    assert first.parameters["random_state"] == 23


@pytest.mark.parametrize("method,params", [
    ("spectral", {"n_clusters": 3, "n_neighbors": 10}),
    ("agglomerative", {"n_clusters": 3}),
    ("dbscan", {"eps": 0.8, "min_samples": 5}),
    ("gaussian_mixture", {"n_clusters": 3}),
])
def test_additional_methods_support_explicit_pca_space(blobs, method, params):
    matrix, truth = blobs
    result = cluster(matrix, method, space="pca", pca_components=2, **params)
    assert result.space == "pca"
    assert result.representation.shape == (90, 2)
    assert result.parameters["pca_components"] == 2
    assert adjusted_rand_score(truth, result.labels) == 1


def test_dbscan_finds_noise_and_cosine_groups():
    rng = np.random.default_rng(12)
    matrix = np.concatenate([
        rng.normal([3, 0], 0.02, (20, 2)),
        rng.normal([0, 3], 0.02, (20, 2)),
        [[-3, 0]],
    ])
    result = cluster(matrix, "dbscan", eps=0.01, min_samples=4, metric="cosine")
    assert result.labels[-1] == -1
    assert adjusted_rand_score([0] * 20 + [1] * 20, result.labels[:-1]) == 1
    assert any("1 of 41 objects as noise" in warning for warning in result.warnings)


def test_dbscan_all_noise_has_actionable_warning():
    result = cluster([[1, 0], [0, 1], [5, 4]], "dbscan", eps=0.01, min_samples=2)
    assert np.all(result.labels == -1)
    assert "every object to noise" in result.warnings[-1]


def test_gaussian_mixture_reports_failed_convergence():
    matrix, _ = make_blobs(n_samples=120, centers=3, cluster_std=3, random_state=20)
    result = cluster(matrix, "gaussian_mixture", n_clusters=3, max_iter=1)
    assert any("converg" in message.lower() for message in result.warnings)
    assert result.labels.shape == (120,)


@pytest.mark.parametrize("method,params,error", [
    ("spectral", {"n_clusters": 4}, "n_clusters"),
    ("spectral", {"n_clusters": True}, "n_clusters"),
    ("spectral", {"n_clusters": 1.5}, "n_clusters"),
    ("spectral", {"n_neighbors": 1}, "n_neighbors"),
    ("spectral", {"n_neighbors": 5}, "n_neighbors"),
    ("spectral", {"n_neighbors": False}, "n_neighbors"),
    ("spectral", {"metric": "cosine"}, "Unsupported"),
    ("agglomerative", {"n_clusters": 0}, "n_clusters"),
    ("agglomerative", {"n_clusters": 5}, "n_clusters"),
    ("agglomerative", {"linkage": "centroid"}, "linkage"),
    ("agglomerative", {"linkage": "ward", "metric": "cosine"}, "Ward linkage"),
    ("agglomerative", {"metric": "dot_product"}, "metric"),
    ("agglomerative", {"eps": 0.5}, "Unsupported"),
    ("dbscan", {"eps": 0}, "eps"),
    ("dbscan", {"eps": -1}, "eps"),
    ("dbscan", {"eps": np.inf}, "eps"),
    ("dbscan", {"eps": np.nan}, "eps"),
    ("dbscan", {"eps": True}, "eps"),
    ("dbscan", {"min_samples": 0}, "min_samples"),
    ("dbscan", {"min_samples": 5}, "min_samples"),
    ("dbscan", {"min_samples": 1.5}, "min_samples"),
    ("dbscan", {"metric": "dot_product"}, "metric"),
    ("dbscan", {"n_clusters": 2}, "Unsupported"),
    ("gaussian_mixture", {"n_clusters": 5}, "n_clusters"),
    ("gaussian_mixture", {"n_clusters": True}, "n_clusters"),
    ("gaussian_mixture", {"covariance_type": "full"}, "covariance_type"),
    ("gaussian_mixture", {"max_iter": 0}, "max_iter"),
    ("gaussian_mixture", {"max_iter": 10_001}, "max_iter"),
    ("gaussian_mixture", {"max_iter": True}, "max_iter"),
    ("gaussian_mixture", {"reg_covar": 0}, "reg_covar"),
    ("gaussian_mixture", {"reg_covar": np.inf}, "reg_covar"),
    ("gaussian_mixture", {"reg_covar": np.nan}, "reg_covar"),
    ("gaussian_mixture", {"reg_covar": False}, "reg_covar"),
    ("gaussian_mixture", {"n_components": 2}, "Unsupported"),
])
def test_invalid_additional_parameters_are_rejected(method, params, error):
    matrix = [[1, 1], [2, 1], [1, 3], [4, 1]]
    with pytest.raises(ComputeError, match=error):
        cluster(matrix, method, **params)


@pytest.mark.parametrize("method", CLUSTER_METHODS)
def test_invalid_random_state_and_nonfinite_vectors_rejected_for_every_method(method):
    with pytest.raises(ComputeError, match="random_state"):
        cluster(np.ones((4, 2)), method, random_state=True)
    with pytest.raises(ComputeError, match="finite"):
        cluster([[1, 1], [np.nan, 2], [2, 1]], method)


@pytest.mark.parametrize("method,params", [
    ("agglomerative", {"linkage": "average"}),
    ("dbscan", {}),
])
def test_cosine_methods_reject_zero_vectors(method, params):
    with pytest.raises(ComputeError, match="zero vectors"):
        cluster([[0, 0], [1, 1], [2, 1]], method, metric="cosine", **params)


@pytest.mark.parametrize("method,rows,error", [
    ("spectral", 1, "at least 3"),
    ("spectral", 2, "at least 3"),
    ("agglomerative", 1, "at least 2"),
    ("gaussian_mixture", 1, "at least 2"),
])
def test_minimal_dataset_errors_are_explicit(method, rows, error):
    with pytest.raises(ComputeError, match=error):
        cluster(np.ones((rows, 2)), method)


def test_minimal_supported_datasets():
    assert list(cluster([[1.0]], "dbscan").labels) == [0]
    assert list(cluster([[1.0], [2.0]], "agglomerative", n_clusters=1).labels) == [0, 0]
    assert list(cluster([[1.0], [2.0]], "gaussian_mixture", n_clusters=1).labels) == [0, 0]
    result = cluster([[1, 0], [0, 1], [2, 1]], "spectral", n_clusters=2, n_neighbors=2)
    assert len(np.unique(result.labels)) == 2


@pytest.mark.parametrize("method,estimator", [
    ("agglomerative", clustering.AgglomerativeClustering),
    ("dbscan", clustering.DBSCAN),
])
def test_quadratic_storage_limit_rejects_before_fit(monkeypatch, method, estimator):
    def forbidden_fit(*args, **kwargs):
        pytest.fail("Estimator must not run outside the memory budget")

    monkeypatch.setattr(estimator, "fit_predict", forbidden_fit)
    with pytest.raises(ComputeError, match="512 MiB.*8,192"):
        cluster(np.ones((8193, 2)), method)


@pytest.mark.parametrize("rows,neighbors,error", [
    (20_001, 10, "20,000"),
    (20_000, 1000, "256 MiB"),
])
def test_spectral_resource_limits_reject_before_fit(monkeypatch, rows, neighbors, error):
    def forbidden_fit(*args, **kwargs):
        pytest.fail("Spectral estimator must not run outside its resource limits")

    monkeypatch.setattr(clustering.SpectralClustering, "fit_predict", forbidden_fit)
    with pytest.raises(ComputeError, match=error):
        cluster(np.ones((rows, 2)), "spectral", n_neighbors=neighbors)


def test_gaussian_mixture_responsibility_limit_rejects_before_fit(monkeypatch):
    def forbidden_fit(*args, **kwargs):
        pytest.fail("EM must not run outside its memory budget")

    monkeypatch.setattr(clustering.GaussianMixture, "fit_predict", forbidden_fit)
    with pytest.raises(ComputeError, match="512 MiB"):
        cluster(np.ones((90_000, 1)), "gaussian_mixture", n_clusters=256)
