"""Clustering methods retain their results across map styling, cache, and studies."""

import time

import numpy as np
import pytest

from embedding_visualizer.data import DatasetManifest, write_dataset
from embedding_visualizer.workspace import Workspace


@pytest.fixture
def cluster_workspace(tmp_path):
    rng = np.random.default_rng(57)
    vectors = np.vstack([
        rng.normal([-3, -2, 1], .15, size=(12, 3)),
        rng.normal([3, 2, -1], .15, size=(12, 3)),
    ]).astype(np.float32)
    ids = [f"object-{23 - index:02}" for index in range(len(vectors))]
    metadata = [{"keep": index % 3 != 1, "score": index * 10,
                 "kind": "even" if index % 2 == 0 else "odd"}
                for index in range(len(vectors))]
    source = write_dataset(
        tmp_path / "source", ids, vectors,
        DatasetManifest("Cluster integration", "fixture", "v1", 3),
        names=[f"Object {index}" for index in range(len(vectors))], metadata=metadata,
    )
    workspace = Workspace(tmp_path / "cache")
    workspace.load(source)
    return workspace


def finish_clustering(workspace, method, params, scope="all"):
    started = workspace.start_operation("clustering", method, params, scope)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with workspace.lock:
            job = workspace.job.copy()
        assert job["id"] == started["id"]
        if job["status"] != "running":
            assert job["status"] == "done", job
            return job
        time.sleep(.005)
    pytest.fail(f"{method} clustering did not complete within 10 seconds")


METHODS = [
    pytest.param("spectral", {"n_clusters": 2, "n_neighbors": 4, "random_state": 7}, id="spectral"),
    pytest.param("gaussian_mixture", {"n_clusters": 2, "covariance_type": "diag", "random_state": 7}, id="gaussian-mixture"),
]


@pytest.mark.parametrize("method,params", METHODS)
def test_new_cluster_study_restores_labels_and_view_without_recomputing(
    cluster_workspace, tmp_path, monkeypatch, method, params,
):
    workspace = cluster_workspace
    workspace.set_filters([{"field": "keep", "op": "in", "values": [True]}])
    finish_clustering(workspace, method, params, scope="slice")
    labels = workspace.clustering["labels"].copy()
    assert len(set(labels[labels >= 0])) == 2
    assert np.all(labels[~workspace.mask] == -2)
    selected = workspace.dataset.ids[2]
    workspace.select([selected], primary=selected)
    view = {
        "theme": "dark", "point_size": 9, "color": "__cluster__", "x": 2, "y": 0,
        "x_range": [-1.25, 3.75], "y_range": [-4, 4],
        "panels": {"settings": False, "inspector": True},
    }
    archive = workspace.save(view)

    def forbidden(*args, **kwargs):
        pytest.fail("Study restore must use saved labels, not recalculate a clustering")

    monkeypatch.setattr("embedding_visualizer.workspace.cluster", forbidden)
    restored = Workspace(tmp_path / "fresh-cache")
    info = restored.restore(archive)

    np.testing.assert_array_equal(restored.clustering["labels"], labels)
    assert info["clustering"]["method"] == method
    assert info["clustering"]["request"] == params
    assert info["clustering"]["scope"] == "slice"
    assert info["summaries"] == workspace.info()["summaries"]
    assert info["selected_ids"] == [selected]
    assert info["primary"] == selected
    assert info["view"] == view
    assert info["filters"] == workspace.info()["filters"]


@pytest.mark.parametrize("method,params", METHODS)
def test_new_cluster_cache_is_reused_by_another_workspace(
    cluster_workspace, tmp_path, monkeypatch, method, params,
):
    workspace = cluster_workspace
    assert finish_clustering(workspace, method, params)["cache_hit"] is False
    labels = workspace.clustering["labels"].copy()
    summaries = workspace.info()["summaries"]

    def forbidden(*args, **kwargs):
        pytest.fail("A saved clustering with the same input must be reused from disk")

    monkeypatch.setattr("embedding_visualizer.workspace.cluster", forbidden)
    reopened = Workspace(tmp_path / "cache")
    reopened.load(workspace.source)
    assert finish_clustering(reopened, method, params)["cache_hit"] is True
    np.testing.assert_array_equal(reopened.clustering["labels"], labels)
    assert reopened.info()["summaries"] == summaries
    assert reopened.clustering["method"] == method


def test_map_cluster_labels_follow_ids_independently_of_metadata_colors(cluster_workspace):
    workspace = cluster_workspace
    assert workspace.map_data()["clusters"] is None
    finish_clustering(workspace, "kmeans", {"n_clusters": 2})
    labels_by_id = dict(zip(workspace.dataset.ids, workspace.clustering["labels"].tolist(), strict=True))
    workspace.set_filters([{"field": "keep", "op": "in", "values": [True]}])
    selected_ids = [identifier for identifier, keep in zip(workspace.dataset.ids, workspace.mask, strict=True) if keep]
    for color in ("", "score", "kind", "__cluster__"):
        points = workspace.map_data(color=color)
        assert points["ids"] == selected_ids
        assert points["clusters"] == [labels_by_id[identifier] for identifier in selected_ids]
        if color == "score":
            assert points["color_kind"] == "numeric"
            assert points["values"] == [workspace.dataset.metadata[workspace.id_index[identifier]]["score"]
                                        for identifier in selected_ids]
        elif color == "__cluster__":
            assert points["values"] == points["clusters"]
    workspace.clear_clusters()
    assert workspace.map_data(color="score")["clusters"] is None


def test_map_marks_objects_outside_the_clustered_slice_even_with_metadata_color(cluster_workspace):
    workspace = cluster_workspace
    workspace.set_filters([{"field": "keep", "op": "in", "values": [True]}])
    computed_mask = workspace.mask.copy()
    finish_clustering(workspace, "kmeans", {"n_clusters": 2}, scope="slice")
    workspace.set_filters([])
    points = workspace.map_data(color="kind")
    assert points["ids"] == workspace.dataset.ids
    assert points["clusters"] == workspace.clustering["labels"].tolist()
    assert np.all(np.asarray(points["clusters"])[~computed_mask] == -2)
    assert np.all(np.asarray(points["clusters"])[computed_mask] >= 0)
    assert points["values"] == [item["kind"] for item in workspace.dataset.metadata]
