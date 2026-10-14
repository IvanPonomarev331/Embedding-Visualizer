"""Exact, single-query neighbor search without a pairwise distance matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .common import ComputeError, prepare_embeddings, validate_matrix


@dataclass(frozen=True)
class NeighborResult:
    indices: np.ndarray
    values: np.ndarray
    metric: str
    value_kind: str
    excluded_zero_count: int = 0


def exact_neighbors(
    embeddings: Any,
    query_index: int,
    k: int = 10,
    metric: str = "cosine",
) -> NeighborResult:
    """Find up to k neighbors, excluding the query and breaking ties by row.

    Cosine and dot product return descending similarities; Euclidean returns
    ascending L2 distances. Cosine excludes zero candidate vectors and rejects
    a zero query. Score computation and storage scale with N×D, never N×N;
    only the selected neighbors need sorting.
    """
    matrix = validate_matrix(embeddings)
    if isinstance(query_index, (bool, np.bool_)) or not isinstance(query_index, (int, np.integer)):
        raise ComputeError("query_index must be an integer row index.")
    if not 0 <= query_index < len(matrix):
        raise ComputeError("query_index is outside the embedding matrix.")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ComputeError("k must be a positive integer.")
    if not isinstance(metric, str) or metric not in ("cosine", "euclidean", "dot_product"):
        raise ComputeError("metric must be cosine, euclidean, or dot_product.")

    indices = np.arange(len(matrix), dtype=np.int64)
    include = indices != query_index
    excluded_zero_count = 0
    if metric == "cosine":
        zero = np.all(matrix == 0, axis=1)
        if zero[query_index]:
            raise ComputeError("Cosine similarity is undefined for a zero query vector. Choose Euclidean or dot product.")
        excluded_zero_count = int(np.count_nonzero(zero & include))
        include &= ~zero
        normalized = prepare_embeddings(matrix, normalize=True)
        values = np.clip(normalized @ normalized[query_index], -1.0, 1.0)
    elif metric == "euclidean":
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.hypot.reduce(matrix - matrix[query_index], axis=1)
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            values = matrix @ matrix[query_index]

    indices = indices[include]
    values = values[include]
    if not np.isfinite(values).all():
        raise ComputeError("Neighbor scores exceed float64 range. Enable L2 normalization or rescale the embeddings.")
    ranks = values if metric == "euclidean" else -values
    if k < len(indices):
        cutoff = np.partition(ranks, k - 1)[k - 1]
        closer = np.flatnonzero(ranks < cutoff)
        # Candidate indices still follow input order, so cutoff ties can be
        # selected directly without sorting the entire dataset.
        tied = np.flatnonzero(ranks == cutoff)[:k - len(closer)]
        selected = np.concatenate((closer, tied))
    else:
        selected = np.arange(len(indices))
    order = selected[np.lexsort((indices[selected], ranks[selected]))]
    return NeighborResult(
        indices=indices[order],
        values=values[order],
        metric=metric,
        value_kind="distance" if metric == "euclidean" else "similarity",
        excluded_zero_count=excluded_zero_count,
    )
