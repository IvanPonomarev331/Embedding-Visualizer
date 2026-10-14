"""Cluster membership, representative IDs, and bounded metadata summaries."""

import numpy as np
import pytest

from embedding_visualizer.compute.common import ComputeError
from embedding_visualizer.compute.summaries import summarize_clusters
from embedding_visualizer.data.models import Dataset, DatasetManifest, ValidationReport


def dataset(embeddings, metadata=None):
    matrix = np.asarray(embeddings, dtype=np.float64)
    count, dimension = matrix.shape
    return Dataset(
        ids=[f"id-{index}" for index in range(count)],
        names=[None] * count,
        embeddings=matrix,
        metadata=metadata if metadata is not None else [{} for _ in range(count)],
        manifest=DatasetManifest("test", "test", "1", dimension),
        report=ValidationReport(count, dimension),
    )


def test_representatives_use_cluster_members_and_given_computation_space():
    data = dataset([[0], [1], [100], [7], [8], [50]])
    summaries = summarize_clusters(data, [0, 0, 0, 1, 1, -1], data.embeddings, max_examples=2)
    assert [(item["cluster_id"], item["size"], item["is_noise"]) for item in summaries] == [
        (-1, 1, True), (0, 3, False), (1, 2, False),
    ]
    assert summaries[0]["example_ids"] == ["id-5"]
    assert summaries[1]["example_ids"] == ["id-1", "id-0"]
    assert summaries[2]["example_ids"] == ["id-3", "id-4"]
    different_space = np.array([[0], [100], [1], [7], [8], [50]])
    recomputed = summarize_clusters(data, [0, 0, 0, 1, 1, -1], different_space, max_examples=1)
    assert recomputed[1]["example_ids"] == ["id-2"]


def test_metadata_numeric_missing_null_nested_and_bounded_values():
    data = dataset([[0], [1], [2], [3]], [
        {"category": "a", "score": 2, "active": True, "nested": {}},
        {"category": "a", "score": 4.0, "active": False, "nested": []},
        {"category": "b", "score": None, "active": True, "nested": None},
        {"category": "c"},
    ])
    summary = summarize_clusters(data, [0, 0, 0, 0], data.embeddings, max_values=1)[0]
    score = summary["metadata"]["score"]
    assert score["type_counts"] == {"integer": 1, "number": 1, "null": 1}
    assert score["missing_count"] == 1
    assert score["null_count"] == 1
    assert score["numeric"] == {"count": 2, "min": 2, "max": 4, "mean": 3}
    category = summary["metadata"]["category"]
    assert category["values"] == [{"value": "a", "type": "string", "count": 2}]
    assert category["other_values_count"] == 2
    active = summary["metadata"]["active"]
    assert active["values"] == [{"value": True, "type": "boolean", "count": 2}]
    assert "numeric" not in active
    assert summary["metadata"]["nested"]["type_counts"] == {"array": 1, "null": 1, "object": 1}


def test_fields_use_same_bounded_selection_in_every_cluster():
    data = dataset([[0], [1]], [{"a": 1}, {"b": 2, "c": 3}])
    results = summarize_clusters(data, [0, 1], data.embeddings, max_fields=2)
    for result in results:
        assert list(result["metadata"]) == ["a", "b"]
        assert result["metadata_fields_omitted"] == 1
    assert results[1]["metadata"]["a"]["missing_count"] == 1


def test_extreme_finite_values_keep_centroid_and_mean_usable():
    data = dataset([[1e308], [1e308]], [{"n": 1e308}, {"n": 1e308}])
    result = summarize_clusters(data, [0, 0], data.embeddings)[0]
    assert result["example_ids"] == ["id-0", "id-1"]
    assert result["metadata"]["n"]["numeric"]["mean"] == 1e308
    data.metadata = [{"n": 10 ** 400}, {"n": 10 ** 400}]
    assert summarize_clusters(data, [0, 0], data.embeddings)[0]["metadata"]["n"]["numeric"]["mean"] is None


@pytest.mark.parametrize("labels", [[0], [0, -2], [0, True], [0.0, 1.0], [[0], [1]], ["0", "1"]])
def test_invalid_labels_rejected(labels):
    data = dataset([[0], [1]])
    with pytest.raises(ComputeError):
        summarize_clusters(data, labels, data.embeddings)


def test_alignment_and_limits_validated():
    data = dataset([[0], [1]])
    with pytest.raises(ComputeError, match="same rows"):
        summarize_clusters(data, [0], [[0]])
    for kwargs in ({"max_examples": 0}, {"max_fields": True}, {"max_values": 0.5}):
        with pytest.raises(ComputeError, match="positive integer"):
            summarize_clusters(data, [0, 0], data.embeddings, **kwargs)
