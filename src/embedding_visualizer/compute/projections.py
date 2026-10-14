"""Validated projection operations, independent of the interface and source data."""

from __future__ import annotations

from dataclasses import dataclass, field
import numbers
import warnings as python_warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

from .common import ComputeError, validate_matrix


@dataclass(frozen=True)
class ProjectionResult:
    coordinates: np.ndarray
    method: str
    parameters: dict
    explained_variance_ratio: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)


def _integer(value, name: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ComputeError(f"{name} must be an integer.")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"between {minimum} and {maximum}" if maximum is not None else f"at least {minimum}"
        raise ComputeError(f"{name} must be {bound}.")
    return value


def _real(value, name: str, minimum: float, maximum: float, *, lower_open=False, upper_open=False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not np.isfinite(value):
        raise ComputeError(f"{name} must be a finite number.")
    value = float(value)
    if (value <= minimum if lower_open else value < minimum) or (value >= maximum if upper_open else value > maximum):
        lower = "greater than" if lower_open else "at least"
        upper = "less than" if upper_open else "at most"
        raise ComputeError(f"{name} must be {lower} {minimum} and {upper} {maximum}.")
    return value


def _metric(metric, matrix: np.ndarray) -> str:
    if metric not in ("euclidean", "cosine"):
        raise ComputeError("metric must be 'euclidean' or 'cosine'.")
    if metric == "cosine" and np.any(np.all(matrix == 0, axis=1)):
        raise ComputeError("Cosine distance is undefined for zero vectors; remove them or choose Euclidean distance.")
    return metric


def _unknown_parameters(params: dict, allowed: set[str], method: str) -> None:
    unknown = params.keys() - allowed
    if unknown:
        raise ComputeError(f"Unsupported {method} parameters: {', '.join(sorted(unknown))}.")


def project(
    embeddings,
    method: str,
    *,
    n_components: int = 2,
    random_state: int = 42,
    **params,
) -> ProjectionResult:
    """Project a finite matrix; row positions remain identical to the input.

    Supported methods are ``pca``, ``umap`` and ``tsne``. PCA accepts no extra
    parameters, UMAP accepts ``n_neighbors``, ``min_dist`` and ``metric``, and
    t-SNE accepts ``perplexity``, ``max_iter`` and ``metric``. The input is
    never normalized implicitly. Nonlinear coordinates are for visualization,
    not for measuring neighbor distances in the original embedding space.
    """
    matrix = validate_matrix(embeddings)
    if method not in ("pca", "umap", "tsne"):
        raise ComputeError("Projection method must be 'pca', 'umap' or 'tsne'.")
    n_rows, n_features = matrix.shape
    if n_rows < 2:
        raise ComputeError("Projection requires at least 2 objects.")
    seed = _integer(random_state, "random_state", 0, 2**32 - 1)
    components = _integer(n_components, "n_components", 1)
    parameters = {"n_components": components, "random_state": seed}
    if method == "pca":
        _unknown_parameters(params, set(), method)
        _integer(components, "n_components", 1, min(n_rows, n_features))
        parameters["svd_solver"] = "full"
        estimator = PCA(n_components=components, svd_solver="full", random_state=seed)
    elif method == "umap":
        _unknown_parameters(params, {"n_neighbors", "min_dist", "metric"}, method)
        if n_rows < 3:
            raise ComputeError("UMAP requires at least 3 objects; use raw coordinates or PCA for smaller datasets.")
        neighbors = _integer(params.get("n_neighbors", min(15, n_rows - 1)), "n_neighbors", 2, n_rows - 1)
        min_dist = _real(params.get("min_dist", 0.1), "min_dist", 0.0, 1.0)
        metric = _metric(params.get("metric", "euclidean"), matrix)
        # Lazy import avoids JIT initialization when the user only needs PCA/kNN.
        from umap import UMAP

        parameters.update(n_neighbors=neighbors, min_dist=min_dist, metric=metric, init="random", n_jobs=1)
        estimator = UMAP(n_components=components, n_neighbors=neighbors, min_dist=min_dist,
                         metric=metric, random_state=seed, init="random", n_jobs=1)
    else:
        _unknown_parameters(params, {"perplexity", "max_iter", "metric"}, method)
        perplexity = _real(params.get("perplexity", min(30.0, n_rows - 1.0)), "perplexity", 0.0, float(n_rows), lower_open=True, upper_open=True)
        max_iter = _integer(params.get("max_iter", 1000), "max_iter", 250)
        metric = _metric(params.get("metric", "euclidean"), matrix)
        solver = "barnes_hut" if components <= 3 else "exact"
        parameters.update(perplexity=perplexity, max_iter=max_iter, metric=metric,
                          init="random", learning_rate="auto", method=solver, n_jobs=1)
        estimator = TSNE(n_components=components, perplexity=perplexity, max_iter=max_iter,
                         metric=metric, random_state=seed, init="random", learning_rate="auto",
                         method=solver, n_jobs=1)
    # Constant data have no variance. sklearn's PCA reports NaN ratios for this
    # case; zero coordinates and zero explained variance are the honest result.
    if method == "pca" and np.all(matrix == matrix[0]):
        return ProjectionResult(np.zeros((n_rows, components)), method, parameters,
                                np.zeros(components), ["All vectors are identical; PCA has no variance to explain."])
    try:
        with python_warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
            python_warnings.simplefilter("always")
            coordinates = estimator.fit_transform(matrix)
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
        raise ComputeError(f"{method.upper()} could not compute a projection: {exc}") from exc
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape != (n_rows, components) or not np.isfinite(coordinates).all():
        raise ComputeError(f"{method.upper()} produced non-finite or invalid coordinates; check input scale and parameters.")
    variance = np.asarray(estimator.explained_variance_ratio_, dtype=np.float64) if method == "pca" else None
    if variance is not None and not np.isfinite(variance).all():
        raise ComputeError("PCA produced non-finite explained variance; check input scale.")
    messages = list(dict.fromkeys(str(item.message) for item in captured))
    return ProjectionResult(coordinates, method, parameters, variance, messages)
