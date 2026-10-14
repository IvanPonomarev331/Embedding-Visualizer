"""Clustering in an explicitly chosen embedding or PCA space."""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings as python_warnings

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, HDBSCAN, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from .common import ComputeError, validate_matrix
from .projections import _integer, _metric, _real, _unknown_parameters, project


# sklearn HDBSCAN brute allocates a dense float64 N x N distance matrix.
# This is a bound on that matrix alone; estimator workspace adds overhead.
_HDBSCAN_PAIRWISE_BUDGET_BYTES = 512 * 1024**2
_PAIRWISE_BUDGET_BYTES = 512 * 1024**2
_SPECTRAL_GRAPH_BUDGET_BYTES = 256 * 1024**2
_SPECTRAL_MAX_ROWS = 20_000
_MIXTURE_WORKSPACE_BUDGET_BYTES = 512 * 1024**2

CLUSTER_METHODS = (
    "kmeans", "hdbscan", "spectral", "agglomerative", "dbscan", "gaussian_mixture",
)


def _pairwise_budget(n_rows: int, method: str) -> None:
    """Bound quadratic methods before sklearn can allocate their workspace.

    Agglomerative linkage stores pairwise distances; DBSCAN may store every
    other row in every neighborhood. Use a conservative full float64 matrix
    equivalent, although the estimators' exact overhead differs.
    """
    pairwise_bytes = n_rows * n_rows * np.dtype(np.float64).itemsize
    if pairwise_bytes > _PAIRWISE_BUDGET_BYTES:
        raise ComputeError(
            f"{method} may need quadratic pairwise storage of "
            f"{pairwise_bytes / 1024**2:.1f} MiB for {n_rows:,} objects, exceeding "
            "the 512 MiB matrix budget (additional workspace is also needed). "
            "Compute on a slice of at most 8,192 objects or choose KMeans."
        )


@dataclass(frozen=True)
class ClusteringResult:
    labels: np.ndarray
    method: str
    parameters: dict
    space: str
    representation: np.ndarray
    warnings: list[str] = field(default_factory=list)


def cluster(
    embeddings,
    method: str,
    *,
    space: str = "original",
    pca_components: int | None = None,
    random_state: int = 42,
    **params,
) -> ClusteringResult:
    """Cluster rows without using a visualization's nonlinear coordinates.

    KMeans, spectral, agglomerative and Gaussian mixture accept ``n_clusters``.
    Spectral uses a sparse Euclidean nearest-neighbor graph (``n_neighbors``).
    Agglomerative accepts ``linkage`` and ``metric``; Ward requires Euclidean.
    DBSCAN accepts ``eps``, ``min_samples`` and ``metric``. HDBSCAN also accepts
    ``min_cluster_size``. Density methods mark noise as -1. Gaussian mixture
    supports diagonal/spherical covariance, ``max_iter`` and ``reg_covar``.
    If ``space='pca'``, PCA is fitted inside this call with ``pca_components``.
    """
    matrix = validate_matrix(embeddings)
    if method not in CLUSTER_METHODS:
        raise ComputeError(f"Clustering method must be one of: {', '.join(CLUSTER_METHODS)}.")
    if space not in ("original", "pca"):
        raise ComputeError("Clustering space must be 'original' or 'pca'.")
    seed = _integer(random_state, "random_state", 0, 2**32 - 1)
    n_rows = len(matrix)
    parameters = {"random_state": seed}
    messages = []
    if space == "pca":
        if pca_components is None:
            raise ComputeError("pca_components is required when clustering in PCA space.")
        pca_result = project(matrix, "pca", n_components=pca_components, random_state=seed)
        matrix = pca_result.coordinates
        parameters["pca_components"] = int(pca_components)
        messages.extend(pca_result.warnings)
    elif pca_components is not None:
        raise ComputeError("pca_components is only valid when space is 'pca'.")
    if method == "kmeans":
        _unknown_parameters(params, {"n_clusters"}, method)
        n_clusters = _integer(params.get("n_clusters", min(8, n_rows)), "n_clusters", 1, n_rows)
        parameters.update(n_clusters=n_clusters, metric="euclidean", n_init=10)
        estimator = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    elif method == "hdbscan":
        _unknown_parameters(params, {"min_cluster_size", "min_samples", "metric"}, method)
        if n_rows < 2:
            raise ComputeError("HDBSCAN requires at least 2 objects.")
        min_cluster_size = _integer(params.get("min_cluster_size", min(5, n_rows)), "min_cluster_size", 2, n_rows)
        min_samples = params.get("min_samples")
        if min_samples is not None:
            min_samples = _integer(min_samples, "min_samples", 1, n_rows)
        metric = _metric(params.get("metric", "euclidean"), matrix)
        algorithm = "brute" if metric == "cosine" else "auto"
        if algorithm == "brute":
            pairwise_bytes = n_rows * n_rows * np.dtype(np.float64).itemsize
            if pairwise_bytes > _HDBSCAN_PAIRWISE_BUDGET_BYTES:
                raise ComputeError(
                    f"HDBSCAN with cosine distance needs a dense pairwise matrix "
                    f"of {pairwise_bytes / 1024**2:.1f} MiB for {n_rows:,} objects, "
                    "exceeding the 512 MiB matrix budget (additional workspace is also needed). "
                    "Choose Euclidean distance or load fewer objects (at most 8,192)."
                )
        parameters.update(min_cluster_size=min_cluster_size, min_samples=min_samples,
                          metric=metric, algorithm=algorithm, n_jobs=1)
        estimator = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                            metric=metric, algorithm=algorithm, n_jobs=1, copy=True)
    elif method == "spectral":
        _unknown_parameters(params, {"n_clusters", "n_neighbors"}, method)
        if n_rows < 3:
            raise ComputeError("Spectral clustering requires at least 3 objects; use KMeans for smaller datasets.")
        if n_rows > _SPECTRAL_MAX_ROWS:
            raise ComputeError("Spectral clustering is limited to 20,000 objects because graph construction and the eigensolver can be slow. Compute on a slice or choose KMeans.")
        # k < N prevents scipy's sparse eigensolver from falling back to an
        # unsupported dense solve. Large eigenspaces are expensive even with
        # sparse affinity, so the cluster count is also explicitly bounded.
        n_clusters = _integer(params.get("n_clusters", min(8, n_rows - 1)), "n_clusters", 1, min(256, n_rows - 1))
        n_neighbors = _integer(params.get("n_neighbors", min(15, n_rows)), "n_neighbors", 2, min(1000, n_rows))
        # Symmetrization may double nonzeros; reserve space for both the graph
        # and its Laplacian, with float64 values and conservative int64 indices.
        graph_bytes = 4 * n_rows * n_neighbors * (8 + 8)
        if graph_bytes > _SPECTRAL_GRAPH_BUDGET_BYTES:
            raise ComputeError("Spectral nearest-neighbor graph exceeds the 256 MiB graph budget. Reduce n_neighbors or compute on a smaller slice.")
        parameters.update(n_clusters=n_clusters, n_neighbors=n_neighbors,
                          affinity="nearest_neighbors", metric="euclidean",
                          eigen_solver="arpack", assign_labels="kmeans", n_init=10, n_jobs=1)
        estimator = SpectralClustering(n_clusters=n_clusters, n_neighbors=n_neighbors,
                                       affinity="nearest_neighbors", eigen_solver="arpack",
                                       assign_labels="kmeans", random_state=seed, n_init=10, n_jobs=1)
    elif method == "agglomerative":
        _unknown_parameters(params, {"n_clusters", "linkage", "metric"}, method)
        if n_rows < 2:
            raise ComputeError("Agglomerative clustering requires at least 2 objects.")
        n_clusters = _integer(params.get("n_clusters", min(8, n_rows)), "n_clusters", 1, n_rows)
        linkage = params.get("linkage", "ward")
        if linkage not in ("ward", "average", "complete", "single"):
            raise ComputeError("linkage must be 'ward', 'average', 'complete' or 'single'.")
        metric = _metric(params.get("metric", "euclidean"), matrix)
        if linkage == "ward" and metric != "euclidean":
            raise ComputeError("Ward linkage requires Euclidean distance; choose average, complete or single linkage for cosine distance.")
        _pairwise_budget(n_rows, "Agglomerative clustering")
        parameters.update(n_clusters=n_clusters, linkage=linkage, metric=metric)
        estimator = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric=metric)
    elif method == "dbscan":
        _unknown_parameters(params, {"eps", "min_samples", "metric"}, method)
        eps = _real(params.get("eps", 0.5), "eps", 0.0, np.finfo(float).max, lower_open=True)
        min_samples = _integer(params.get("min_samples", min(5, n_rows)), "min_samples", 1, n_rows)
        metric = _metric(params.get("metric", "euclidean"), matrix)
        _pairwise_budget(n_rows, "DBSCAN")
        parameters.update(eps=eps, min_samples=min_samples, metric=metric, n_jobs=1)
        estimator = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=1)
    else:
        _unknown_parameters(params, {"n_clusters", "covariance_type", "max_iter", "reg_covar"}, method)
        if n_rows < 2:
            raise ComputeError("Gaussian mixture requires at least 2 objects.")
        n_clusters = _integer(params.get("n_clusters", min(8, n_rows)), "n_clusters", 1, min(256, n_rows))
        covariance_type = params.get("covariance_type", "diag")
        if covariance_type not in ("diag", "spherical"):
            raise ComputeError("covariance_type must be 'diag' or 'spherical'.")
        max_iter = _integer(params.get("max_iter", 200), "max_iter", 1, 10_000)
        reg_covar = _real(params.get("reg_covar", 1e-6), "reg_covar", 0.0, np.finfo(float).max, lower_open=True)
        # EM holds multiple N x K responsibility/log-probability arrays.
        if 3 * n_rows * n_clusters * 8 > _MIXTURE_WORKSPACE_BUDGET_BYTES:
            raise ComputeError("Gaussian mixture exceeds the 512 MiB responsibility workspace budget. Reduce n_clusters or compute on a slice.")
        parameters.update(n_clusters=n_clusters, covariance_type=covariance_type,
                          max_iter=max_iter, reg_covar=reg_covar, n_init=1)
        estimator = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type,
                                    max_iter=max_iter, reg_covar=reg_covar, random_state=seed, n_init=1)
    try:
        with python_warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
            python_warnings.simplefilter("always")
            labels = estimator.fit_predict(matrix)
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
        raise ComputeError(f"{method.upper()} could not compute clusters: {exc}") from exc
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (n_rows,) or np.any(labels < -1):
        raise ComputeError(f"{method.upper()} produced invalid cluster labels.")
    messages.extend(str(item.message) for item in captured)
    noise_count = int(np.count_nonzero(labels == -1))
    if noise_count == n_rows:
        messages.append(f"{method.upper()} assigned every object to noise; adjust its parameters or choose another space.")
    elif noise_count:
        noise_percent = 100 * noise_count / n_rows
        percentage = f"{noise_percent:.1f}" if noise_percent >= 0.1 else "<0.1"
        messages.append(f"{method.upper()} marked {noise_count:,} of {n_rows:,} objects as noise ({percentage}%).")
    if "n_clusters" in parameters:
        actual_clusters = len(np.unique(labels[labels >= 0]))
        if actual_clusters < parameters["n_clusters"]:
            messages.append(f"Only {actual_clusters} of {parameters['n_clusters']} requested clusters contain objects; check for duplicate vectors or reduce n_clusters.")
    return ClusteringResult(labels, method, parameters, space, matrix, list(dict.fromkeys(messages)))
