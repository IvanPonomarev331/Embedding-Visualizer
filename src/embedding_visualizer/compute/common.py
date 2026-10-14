"""Validation and explicit preprocessing shared by compute operations."""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np


class ComputeError(ValueError):
    """Invalid input or a computation that cannot produce a usable result."""


def validate_matrix(embeddings: Any) -> np.ndarray:
    """Return a finite, nonempty 2D real matrix in float64.

    Strings, booleans, complex values and object arrays are rejected rather
    than silently converted. A float64 input may be returned without copying;
    callers must treat the result as read-only.
    """
    try:
        matrix = np.asarray(embeddings)
    except (TypeError, ValueError) as error:
        raise ComputeError("Embeddings must be a rectangular 2D numeric matrix.") from error
    if matrix.ndim != 2 or not all(matrix.shape):
        raise ComputeError("Embeddings must be a nonempty 2D matrix.")
    if matrix.dtype.kind not in "iuf":
        raise ComputeError("Embeddings must contain real numbers, not booleans, strings, objects or complex values.")
    # NumPy coerces mixed Python lists such as [1, True] to integers. Inspect
    # their original elements so that the public boundary does not accept them.
    if not isinstance(embeddings, np.ndarray):
        original = np.asarray(embeddings, dtype=object)
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
            for value in original.flat
        ):
            raise ComputeError("Embeddings must contain real numbers, not booleans or coerced values.")
    with np.errstate(over="ignore", invalid="ignore"):
        matrix = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ComputeError("Embeddings must contain only finite numbers representable in float64.")
    return matrix


def prepare_embeddings(embeddings: Any, normalize: bool = False) -> np.ndarray:
    """Return an independent float64 matrix, optionally L2-normalizing rows.

    Scaling before taking a norm avoids overflow and underflow even for very
    large or tiny finite vectors. Zero vectors stay zero; cosine operations
    handle them explicitly because their cosine similarity is undefined.
    """
    if not isinstance(normalize, (bool, np.bool_)):
        raise ComputeError("normalize must be a boolean.")
    matrix = validate_matrix(embeddings).copy()
    if normalize:
        scale = np.max(np.abs(matrix), axis=1)
        nonzero = scale != 0
        matrix[nonzero] /= scale[nonzero, None]
        norms = np.sqrt(np.sum(matrix[nonzero] ** 2, axis=1))
        matrix[nonzero] /= norms[:, None]
    return matrix
