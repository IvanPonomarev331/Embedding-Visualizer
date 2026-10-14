"""Bounded, dataset-agnostic summaries of clusters and their metadata."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

import numpy as np

from embedding_visualizer.data.models import Dataset

from .common import ComputeError, validate_matrix


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _metadata_summary(rows: list[dict[str, Any]], fields: list[str], max_values: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in fields:
        observed = [row[field] for row in rows if field in row]
        types = Counter(_json_type(value) for value in observed)
        summary: dict[str, Any] = {
            "type_counts": dict(sorted(types.items())),
            "missing_count": len(rows) - len(observed),
            "null_count": types.get("null", 0),
        }
        numbers = [value for value in observed if type(value) in (int, float)]
        if numbers:
            # Divide before adding to avoid overflowing the sum. Unusually
            # large JSON integers may still have no representable float mean.
            try:
                mean = math.fsum(value / len(numbers) for value in numbers)
                if not math.isfinite(mean):
                    mean = None
            except (OverflowError, ValueError):
                mean = None
            summary["numeric"] = {
                "count": len(numbers), "min": min(numbers), "max": max(numbers), "mean": mean,
            }
        counts = Counter(
            (_json_type(value), value)
            for value in observed if isinstance(value, (str, bool))
        )
        ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0][0], str(pair[0][1])))
        summary["values"] = [
            {"value": value, "type": kind, "count": count}
            for (kind, value), count in ordered[:max_values]
        ]
        summary["other_values_count"] = sum(count for _, count in ordered[max_values:])
        result[field] = summary
    return result


def summarize_clusters(
    dataset: Dataset,
    labels: np.ndarray,
    embeddings: Any,
    max_examples: int = 5,
    max_fields: int = 12,
    max_values: int = 8,
) -> list[dict[str, Any]]:
    """Summarize memberships in the supplied computation space.

    Each result has cluster_id, size, is_noise, example_ids, metadata and
    metadata_fields_omitted. Results are sorted by label (noise -1 first).
    Representatives are actual members nearest the Euclidean centroid; noise
    examples use row order. Metadata reports top-level type/missing/null counts,
    numeric min/max/mean, and bounded string/boolean frequency tables. Nested
    structures contribute type counts only. Numeric mean is null if it cannot
    be represented as a finite float. Field names are alphabetically bounded
    across the full dataset so the same fields are comparable between clusters.
    """
    matrix = validate_matrix(embeddings)
    count = len(matrix)
    if len(dataset.ids) != count or len(dataset.metadata) != count:
        raise ComputeError("Dataset, labels and embeddings must use the same rows in the same order.")
    for name, value in (("max_examples", max_examples), ("max_fields", max_fields), ("max_values", max_values)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ComputeError(f"{name} must be a positive integer.")
    try:
        labels_array = np.asarray(labels)
    except (ValueError, TypeError) as error:
        raise ComputeError("labels must contain one integer cluster ID per row.") from error
    if labels_array.ndim != 1 or len(labels_array) != count or labels_array.dtype.kind not in "iu":
        raise ComputeError("labels must contain one integer cluster ID per row.")
    if not isinstance(labels, np.ndarray) and any(
        isinstance(value, (bool, np.bool_)) for value in np.asarray(labels, dtype=object)
    ):
        raise ComputeError("labels must contain integer cluster IDs, not booleans.")
    if np.any(labels_array < -1):
        raise ComputeError("Cluster IDs must be nonnegative, with -1 reserved for noise.")

    all_fields = sorted({field for row in dataset.metadata for field in row})
    fields = all_fields[:max_fields]
    summaries: list[dict[str, Any]] = []
    by_label = np.argsort(labels_array, kind="stable")
    boundaries = np.flatnonzero(labels_array[by_label][1:] != labels_array[by_label][:-1]) + 1
    for members in np.split(by_label, boundaries):
        label = labels_array[members[0]]
        if label == -1:
            examples = members[:max_examples]
        else:
            values = matrix[members]
            scale = np.max(np.abs(values))
            scaled = values / scale if scale else values
            distances = np.hypot.reduce(scaled - np.mean(scaled, axis=0), axis=1)
            order = np.lexsort((members, distances))
            examples = members[order[:max_examples]]
        summaries.append({
            "cluster_id": int(label),
            "size": len(members),
            "is_noise": bool(label == -1),
            "example_ids": [dataset.ids[index] for index in examples],
            "metadata": _metadata_summary([dataset.metadata[index] for index in members], fields, max_values),
            "metadata_fields_omitted": max(0, len(all_fields) - max_fields),
        })
    return summaries
