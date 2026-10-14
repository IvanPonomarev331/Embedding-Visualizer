"""Bounded, reproducible parameter searches in one fixed clustering space."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import math
import time
from typing import Callable

import numpy as np
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

from .clustering import ClusteringResult, cluster
from .common import ComputeError, validate_matrix
from .projections import _integer, project


TUNABLE_METHODS = ("kmeans", "spectral", "agglomerative", "gaussian_mixture")
SEARCH_VERSION = 1
_SILHOUETTE_SAMPLE_SIZE = 1000


class TuningCancelled(ComputeError):
    """A search stopped without publishing its partial best result."""


@dataclass(frozen=True)
class AutoClusteringResult:
    result: ClusteringResult
    report: dict
    request: dict


def _check_cancelled(cancelled: Callable | None) -> None:
    if cancelled is not None and cancelled():
        raise TuningCancelled("Automatic clustering search was cancelled; no result was applied.")


def _seed_at(base: int, index: int) -> int:
    # An odd increment visits all uint32 seeds without duplicates. It does not
    # depend on a process-global RNG or the order of earlier computations.
    return (base + index * 0x9E3779B9) % 2**32


def _candidates(method: str, count: int, rows: int, seed: int, budget: int) -> list[dict]:
    if method == "kmeans":
        configurations = [{}]
    elif method == "spectral":
        # The 256 MiB bound mirrors cluster()'s sparse graph + Laplacian budget.
        # Avoid spending trials on graph sizes known to be inadmissible.
        maximum = min(rows, 1000, (256 * 1024**2) // (64 * rows))
        neighbors = []
        for value in (15, 5, 10, 20, 30, 50, 100, 200, maximum, 2):
            value = min(maximum, value)
            if value >= 2 and value not in neighbors:
                neighbors.append(value)
        configurations = [{"n_neighbors": value} for value in neighbors]
    elif method == "agglomerative":
        configurations = [
            {"linkage": "ward", "metric": "euclidean"},
            {"linkage": "average", "metric": "euclidean"},
            {"linkage": "complete", "metric": "euclidean"},
            {"linkage": "single", "metric": "euclidean"},
            {"linkage": "average", "metric": "cosine"},
            {"linkage": "complete", "metric": "cosine"},
            {"linkage": "single", "metric": "cosine"},
        ]
    else:
        configurations = [
            {"covariance_type": covariance, "reg_covar": regularization, "max_iter": 200}
            for regularization in (1e-6, 1e-4, 1e-2)
            for covariance in ("diag", "spherical")
        ]
    candidates = []
    # Deterministic agglomerative linkage has only seven useful configurations;
    # changing its seed would repeat the same computation.
    rounds = 1 if method == "agglomerative" else budget
    for start in range(rounds):
        for configuration in configurations:
            candidates.append({"n_clusters": count, "random_state": _seed_at(seed, start), **configuration})
            if len(candidates) == budget:
                return candidates
    return candidates


def _inertia(matrix: np.ndarray, labels: np.ndarray) -> float:
    """Exact full-data SSE around empirical centroids, without mean overflow.

    Center each feature at its own scale so constant 1e308-valued columns do
    not overflow their means or erase variation in smaller-valued columns.
    Scale the residual norm too; a genuinely unrepresentable SSE is rejected.
    """
    contributions = []
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for label in np.unique(labels):
            members = matrix[labels == label]
            scales = np.max(np.abs(members), axis=0)
            scaled = np.divide(members, scales, out=np.zeros_like(members), where=scales != 0)
            centroid = np.mean(scaled, axis=0) * scales
            residuals = members - centroid
            magnitude = float(np.max(np.abs(residuals)))
            if magnitude == 0:
                contributions.append(0.0)
                continue
            residuals /= magnitude
            square_sum = float(np.einsum("ij,ij->", residuals, residuals))
            contributions.append((magnitude * square_sum) * magnitude)
        score = math.fsum(contributions)
    if not math.isfinite(score):
        raise ComputeError("Within-cluster SSE is not finite; rescale embeddings before tuning.")
    return score


def tune_clustering(
    embeddings,
    method: str,
    *,
    n_clusters,
    space: str = "original",
    pca_components: int | None = None,
    random_state: int = 42,
    objective: str = "inertia",
    budget: int = 12,
    progress: Callable[[dict], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> AutoClusteringResult:
    """Choose parameters for a selected method and fixed occupied cluster count.

    All trials fit every row in the same representation. Inertia minimizes
    exact full-data Euclidean SSE. Silhouette maximizes Euclidean silhouette
    on one deterministic shared sample of at most 1,000 rows. It never tunes
    PCA dimensions, changes K, or accepts noise/empty clusters. A successful
    baseline is the first candidate and can only be improved upon.

    Cancellation is cooperative between fits/scoring operations. The returned
    request contains only kwargs accepted by manual ``cluster``. PCA currently
    uses deterministic full SVD, so the winning clustering seed reproduces
    the same fixed PCA even when it differs from the representation seed.
    """
    if method not in TUNABLE_METHODS:
        if method in ("dbscan", "hdbscan"):
            raise ComputeError("DBSCAN and HDBSCAN do not specify an exact cluster count; use manual mode for these methods.")
        raise ComputeError(f"Automatic clustering method must be one of: {', '.join(TUNABLE_METHODS)}.")
    matrix = validate_matrix(embeddings)
    rows, dimensions = matrix.shape
    seed = _integer(random_state, "random_state", 0, 2**32 - 1)
    budget = _integer(budget, "budget", 1, 48)
    count = _integer(n_clusters, "n_clusters", 1, rows)
    if objective not in ("inertia", "silhouette"):
        raise ComputeError("objective must be 'inertia' or 'silhouette'.")
    if objective == "silhouette" and not 2 <= count < rows:
        raise ComputeError("Silhouette requires n_clusters between 2 and the number of objects minus 1.")
    if space not in ("original", "pca"):
        raise ComputeError("Clustering space must be 'original' or 'pca'.")
    if progress is not None and not callable(progress):
        raise ComputeError("progress must be a callable or None.")
    if cancelled is not None and not callable(cancelled):
        raise ComputeError("cancelled must be a callable or None.")
    if method == "spectral":
        if rows < 3:
            raise ComputeError("Spectral clustering requires at least 3 objects.")
        if rows > 20_000:
            raise ComputeError("Spectral clustering is limited to 20,000 objects; compute on a slice or choose KMeans.")
        _integer(count, "n_clusters", 1, min(256, rows - 1))
    if method in ("agglomerative", "gaussian_mixture") and rows < 2:
        raise ComputeError(f"{method} requires at least 2 objects.")
    if method == "gaussian_mixture":
        _integer(count, "n_clusters", 1, min(256, rows))
    if space == "pca":
        if pca_components is None:
            raise ComputeError("pca_components is required when clustering in PCA space.")
        pca_components = _integer(pca_components, "pca_components", 1, min(rows, dimensions))
    elif pca_components is not None:
        raise ComputeError("pca_components is only valid when space is 'pca'.")
    _check_cancelled(cancelled)
    configurations = _candidates(method, count, rows, seed, budget)
    total = len(configurations)
    if progress is not None:
        progress({"completed": 0, "total": total, "best_score": None, "trial": None})
    _check_cancelled(cancelled)
    representation_warnings = []
    representation_parameters = {}
    if space == "pca":
        projected = project(matrix, "pca", n_components=pca_components, random_state=seed)
        matrix = projected.coordinates
        representation_warnings = projected.warnings
        representation_parameters = projected.parameters
    _check_cancelled(cancelled)
    evaluation_indices = np.arange(rows)
    if objective == "silhouette" and rows > _SILHOUETTE_SAMPLE_SIZE:
        evaluation_indices = np.sort(np.random.default_rng(seed).choice(rows, _SILHOUETTE_SAMPLE_SIZE, replace=False))
    evaluation_matrix = matrix[evaluation_indices] if objective == "silhouette" else None
    report = {
        "search_version": SEARCH_VERSION,
        "objective": objective, "budget": budget, "total": total, "completed": 0,
        "sample_count": int(len(evaluation_indices)), "score_space": space, "score_metric": "euclidean",
        "base_seed": seed, "representation_seed": seed,
        "representation_parameters": representation_parameters,
        "best_trial": None, "best_score": None, "trials": [],
    }
    best = None
    best_request = None
    for trial_number, configuration in enumerate(configurations, start=1):
        _check_cancelled(cancelled)
        request = {**configuration, "space": space}
        if space == "pca":
            request["pca_components"] = pca_components
        trial = {"trial": trial_number, "params": request.copy(), "parameters": {},
                 "status": "error", "score": None, "actual_clusters": None, "elapsed_seconds": 0.0}
        start = time.perf_counter()
        try:
            # PCA has already been fitted once; trial seeds affect clustering
            # only, and the input representation remains fixed for fair scores.
            result = cluster(matrix, method, **configuration)
            _check_cancelled(cancelled)
            applied_parameters = dict(result.parameters)
            if space == "pca":
                applied_parameters["pca_components"] = pca_components
            trial["parameters"] = applied_parameters
            trial["warnings"] = list(dict.fromkeys(representation_warnings + result.warnings))
            labels = np.asarray(result.labels)
            if labels.shape != (rows,) or labels.dtype.kind not in "iu":
                raise ComputeError("The candidate returned invalid cluster labels.")
            occupied = np.unique(labels[labels >= 0])
            trial["actual_clusters"] = int(len(occupied))
            if np.any(labels < 0) or len(occupied) != count:
                raise ComputeError(f"The candidate produced {len(occupied)} occupied clusters; exactly {count} clusters and no noise are required.")
            with threadpool_limits(limits=1):
                if objective == "inertia":
                    score = _inertia(matrix, labels)
                else:
                    sampled_labels = labels[evaluation_indices]
                    if not 2 <= len(np.unique(sampled_labels)) < len(sampled_labels):
                        raise ComputeError("Silhouette is undefined for this candidate on the fixed evaluation sample; use inertia or a different cluster count.")
                    with np.errstate(over="raise", invalid="raise"):
                        score = float(silhouette_score(evaluation_matrix, sampled_labels, metric="euclidean"))
            if not math.isfinite(score):
                raise ComputeError("The candidate produced a non-finite objective score.")
            _check_cancelled(cancelled)
            trial.update(status="ok", score=score)
            improves = best is None or (score < report["best_score"] if objective == "inertia" else score > report["best_score"])
            if improves:
                best = replace(result, labels=labels.copy(), parameters=applied_parameters, space=space,
                               representation=matrix, warnings=trial["warnings"].copy())
                best_request = request.copy()
                report.update(best_trial=trial_number, best_score=score)
        except TuningCancelled:
            raise
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
            _check_cancelled(cancelled)
            trial["error"] = str(error)
        trial["elapsed_seconds"] = round(time.perf_counter() - start, 6)
        report["trials"].append(trial)
        report["completed"] = trial_number
        if progress is not None:
            # Hand observers independent dictionaries; they must not be able
            # to mutate the winning request or this report through callbacks.
            progress({"completed": trial_number, "total": total,
                      "best_score": report["best_score"], "trial": copy.deepcopy(trial)})
        _check_cancelled(cancelled)
    _check_cancelled(cancelled)
    if best is None:
        detail = report["trials"][0].get("error", "No valid candidate.") if report["trials"] else "No admissible candidates."
        error = ComputeError(f"Automatic clustering found no valid result with exactly {count} clusters. {detail}")
        # Callers may preserve the failed search's diagnostics without applying
        # a clustering result. No partial success is returned on cancellation.
        error.report = report
        raise error
    return AutoClusteringResult(best, report, best_request)
