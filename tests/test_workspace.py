"""File-backed integration of slices, computations, cache, and research state."""

import csv
import io
import threading
import time

import numpy as np
import pyarrow.parquet as pq
import pytest

from embedding_visualizer.data import DatasetManifest, write_dataset
from embedding_visualizer.persistence import read_study_bytes, study_bytes
from embedding_visualizer.workspace import Workspace


IDS = ["z-last", "007", "b", "outside", "猫", "e"]
VECTORS = np.array([[1, 0, 0], [1.1, .1, 0], [1.2, 0, .1],
                    [10, 5, 0], [10.1, 5, .1], [10.2, 5.1, 0]], dtype=np.float32)
NAMES = ["Alpha zero", "Beta", "Alpha three", "Far", "Кошка", None]
METADATA = [{"group": "A" if index % 2 == 0 else "B", "price": index + 1, "active": index < 3}
            for index in range(len(IDS))]
A_FILTER = [{"field": "group", "op": "in", "values": ["A"]}]
B_FILTER = [{"field": "group", "op": "in", "values": ["B"]}]


@pytest.fixture
def source(tmp_path):
    return write_dataset(tmp_path / "source", IDS, VECTORS,
                         DatasetManifest("Integration", "embedding/model", "v1", 3),
                         names=NAMES, metadata=METADATA)


@pytest.fixture
def workspace(tmp_path, source):
    session = Workspace(tmp_path / "cache")
    session.load(source)
    return session


def wait_job(session, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with session.lock:
            job = None if session.job is None else session.job.copy()
        if job and job["status"] != "running":
            return job
        time.sleep(.005)
    pytest.fail("Background computation did not finish before the test deadline")


def compute(session, operation="projection", method="pca", params=None, scope="all"):
    started = session.start_operation(operation, method, params if params is not None else {"n_components": 2}, scope)
    assert started["status"] in ("running", "done")
    finished = wait_job(session)
    assert finished["status"] == "done", finished
    return finished


def test_file_load_filters_search_and_pagination_preserve_explicit_ids(workspace, source):
    info = workspace.info()
    assert info["source"] == str(source.resolve())
    assert info["count"] == 6 and info["visible_count"] == 6
    assert info["primary"] == "z-last"
    assert {item["name"]: item["kind"] for item in info["fields"]} == {
        "active": "boolean", "group": "categorical", "price": "numeric"}
    workspace.set_filters(A_FILTER)
    result = workspace.objects(query="alpha", offset=1, limit=1)
    assert result["total"] == 2
    assert [row["id"] for row in result["rows"]] == ["b"]
    workspace.select(["007"], primary="007")
    assert workspace.info()["primary"] == "007"
    assert [row["id"] for row in workspace.objects()["rows"]] == ["z-last", "b", "猫"]
    assert workspace.object_detail("007", metric="euclidean")["in_slice"] is False


def test_failed_dataset_load_and_invalid_filter_leave_working_session_intact(workspace, tmp_path):
    workspace.set_filters(A_FILTER)
    workspace.select(["b"], primary="b")
    before = workspace.info()
    with pytest.raises((ValueError, OSError)):
        workspace.load(tmp_path / "does-not-exist")
    assert workspace.info() == before
    with pytest.raises(ValueError):
        workspace.set_filters([{"field": "unknown", "op": "exists"}])
    assert workspace.info() == before


def test_projection_on_noncontiguous_slice_keeps_row_alignment_and_missing_rows(workspace):
    workspace.set_filters(A_FILTER)
    compute(workspace, scope="slice")
    coordinates = workspace.coordinates.copy()
    assert coordinates.shape == (6, 2)
    assert np.isfinite(coordinates[[0, 2, 4]]).all()
    assert np.isnan(coordinates[[1, 3, 5]]).all()
    assert workspace.projection["scope"] == "slice"
    assert workspace.projection["count"] == 3
    workspace.set_filters([])
    points = workspace.map_data()
    assert points["ids"] == ["z-last", "b", "猫"]
    assert points["filtered"] == 6 and points["available"] == 3
    np.testing.assert_array_equal(workspace.coordinates, coordinates)


def test_slice_clusters_and_summary_examples_use_global_ids(workspace):
    workspace.set_filters(A_FILTER)
    raw_geometry = workspace.geometry
    compute(workspace, "clustering", "kmeans", {"n_clusters": 2}, scope="slice")
    labels = workspace.clustering["labels"]
    assert labels.shape == (6,)
    assert np.all(labels[[1, 3, 5]] == -2)
    assert np.all(labels[[0, 2, 4]] >= 0)
    assert sum(item["size"] for item in workspace.summaries) == 3
    examples = {identifier for item in workspace.summaries for identifier in item["example_ids"]}
    assert examples == {"z-last", "b", "猫"}
    assert workspace.row(1)["cluster"] is None
    assert workspace.geometry == raw_geometry


def test_neighbors_query_outside_slice_searches_requested_universe(workspace):
    workspace.set_filters(A_FILTER)
    query_index = 1
    full = workspace.object_detail("007", k=100, metric="euclidean", scope="all")
    subset = workspace.object_detail("007", k=100, metric="euclidean", scope="slice")
    expected = sorted([0, 2, 4], key=lambda index: np.linalg.norm(VECTORS[index] - VECTORS[query_index]))
    assert {row["id"] for row in full["neighbors"]} == set(IDS) - {"007"}
    assert [row["id"] for row in subset["neighbors"]] == [IDS[index] for index in expected]
    np.testing.assert_allclose([row["value"] for row in subset["neighbors"]],
                               [np.linalg.norm(VECTORS[index].astype(float) - VECTORS[1]) for index in expected])
    assert full["candidate_count"] == 5
    assert subset["candidate_count"] == 3 and subset["in_slice"] is False
    workspace.set_filters([{"field": "group", "op": "in", "values": []}])
    empty = workspace.object_detail("007", k=5, metric="euclidean", scope="slice")
    assert empty["neighbors"] == [] and empty["candidate_count"] == 0


def test_neighbors_are_independent_of_projection_map(workspace):
    baseline = workspace.object_detail("007", k=4, metric="euclidean")
    compute(workspace, params={"n_components": 1})
    after = workspace.object_detail("007", k=4, metric="euclidean")
    assert after["neighbors"] == baseline["neighbors"]
    assert after["vector"] == baseline["vector"]
    assert workspace.coordinates.shape[1] == 1


def test_async_job_uses_slice_snapshot_while_filter_and_selection_remain_responsive(workspace, monkeypatch):
    import embedding_visualizer.workspace as module

    entered, release = threading.Event(), threading.Event()
    real_project = module.project

    def delayed_project(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return real_project(*args, **kwargs)

    monkeypatch.setattr(module, "project", delayed_project)
    workspace.set_filters(A_FILTER)
    started = workspace.start_operation("projection", "pca", {"n_components": 2}, "slice")
    try:
        assert started["status"] == "running"
        assert entered.wait(2)
        workspace.set_filters(B_FILTER)
        workspace.select(["007"], primary="007")
        assert workspace.info()["visible_count"] == 3
        with pytest.raises(ValueError, match="текущего вычисления"):
            workspace.set_normalization(True)
        with pytest.raises(ValueError, match="текущего вычисления"):
            workspace.start_operation("projection", "pca", {"n_components": 1})
        with pytest.raises(ValueError, match="текущего вычисления"):
            workspace.save({})
    finally:
        release.set()
    assert wait_job(workspace)["status"] == "done"
    assert np.isfinite(workspace.coordinates[[0, 2, 4]]).all()
    assert np.isnan(workspace.coordinates[[1, 3, 5]]).all()
    assert workspace.info()["filters"] == B_FILTER
    assert workspace.info()["primary"] == "007"
    assert workspace.map_data()["shown"] == 0


def test_cache_hits_and_invalidation_by_parameters_normalization_and_source(workspace, source, monkeypatch):
    import embedding_visualizer.workspace as module

    real_project = module.project
    calls = []

    def counted_project(*args, **kwargs):
        calls.append(kwargs.copy())
        return real_project(*args, **kwargs)

    monkeypatch.setattr(module, "project", counted_project)
    assert compute(workspace)["cache_hit"] is False
    original_geometry = workspace.geometry
    assert compute(workspace)["cache_hit"] is True
    assert len(calls) == 1 and workspace.geometry == original_geometry
    assert compute(workspace, params={"n_components": 1})["cache_hit"] is False
    workspace.set_normalization(True)
    assert compute(workspace)["cache_hit"] is False
    old_fingerprint = workspace.fingerprint
    changed_names = ["Renamed", *NAMES[1:]]
    write_dataset(source, IDS, VECTORS, workspace.dataset.manifest, names=changed_names,
                  metadata=METADATA, overwrite=True)
    workspace.load(source)
    assert workspace.fingerprint != old_fingerprint
    assert compute(workspace)["cache_hit"] is False
    assert len(calls) == 4


def test_slice_cache_depends_on_selected_row_ids_not_only_filter_description(workspace):
    workspace.set_filters(A_FILTER)
    assert compute(workspace, scope="slice")["cache_hit"] is False
    first_key = workspace.projection["key"]
    workspace.set_filters(B_FILTER)
    assert compute(workspace, scope="slice")["cache_hit"] is False
    assert workspace.projection["key"] != first_key
    workspace.set_filters(A_FILTER)
    assert compute(workspace, scope="slice")["cache_hit"] is True
    assert workspace.projection["key"] == first_key


def test_failed_compute_and_empty_slice_preserve_last_valid_result(workspace):
    compute(workspace)
    coordinates = workspace.coordinates.copy()
    geometry = workspace.geometry
    workspace.start_operation("projection", "pca", {"n_components": 10})
    failed = wait_job(workspace)
    assert failed["status"] == "error" and failed["error"]
    np.testing.assert_array_equal(workspace.coordinates, coordinates)
    assert workspace.geometry == geometry
    workspace.set_filters([{"field": "group", "op": "in", "values": []}])
    with pytest.raises(ValueError, match="срез пуст"):
        workspace.start_operation("projection", "pca", {"n_components": 2}, "slice")
    np.testing.assert_array_equal(workspace.coordinates, coordinates)


def test_geometry_and_coordinates_stable_under_selection_filters_color_and_cluster_clear(workspace):
    compute(workspace)
    initial = workspace.map_data()
    original = {identifier: (x, y) for identifier, x, y in zip(initial["ids"], initial["x"], initial["y"])}
    workspace.select(["b", "猫"], primary="猫")
    workspace.set_filters(A_FILTER)
    for field in ("", "price", "group"):
        points = workspace.map_data(color=field, include_id="猫")
        assert points["geometry"] == initial["geometry"]
        assert {identifier: (x, y) for identifier, x, y in zip(points["ids"], points["x"], points["y"])} == {
            identifier: original[identifier] for identifier in ("z-last", "b", "猫")}
    compute(workspace, "clustering", "kmeans", {"n_clusters": 2})
    assert workspace.map_data(color="__cluster__")["geometry"] == initial["geometry"]
    workspace.clear_clusters()
    assert workspace.geometry == initial["geometry"]


def test_normalization_resets_computations_but_keeps_selection_and_filter(workspace):
    workspace.set_filters(A_FILTER)
    workspace.select(["b"], primary="b")
    compute(workspace)
    compute(workspace, "clustering", "kmeans", {"n_clusters": 2})
    geometry = workspace.geometry
    workspace.set_normalization(True)
    assert workspace.projection is None and workspace.clustering is None and workspace.summaries == []
    assert workspace.primary == "b" and workspace.filters == A_FILTER
    np.testing.assert_allclose(np.linalg.norm(workspace.matrix, axis=1), 1)
    assert workspace.geometry != geometry
    compute(workspace)
    projected = workspace.coordinates.copy()
    workspace.set_normalization(True)
    np.testing.assert_array_equal(workspace.coordinates, projected)


def test_study_restore_preserves_actual_results_and_settings_without_rerunning_algorithms(workspace, tmp_path, monkeypatch):
    import embedding_visualizer.workspace as module

    workspace.set_normalization(True)
    workspace.set_filters(A_FILTER)
    compute(workspace, scope="slice")
    compute(workspace, "clustering", "kmeans", {"n_clusters": 2, "random_state": 7}, scope="slice")
    workspace.select(["b", "猫"], primary="猫")
    view = {"x": 1, "y": 0, "color": "group", "x_range": [-2, 2], "y_range": [-1, 1]}
    archive = workspace.save(view)
    expected_coordinates = workspace.coordinates.copy()
    expected_labels = workspace.clustering["labels"].copy()

    def forbidden(*args, **kwargs):
        pytest.fail("Restoring saved results must not rerun a numerical algorithm")

    monkeypatch.setattr(module, "project", forbidden)
    monkeypatch.setattr(module, "cluster", forbidden)
    restored = Workspace(tmp_path / "empty-cache")
    info = restored.restore(archive)
    np.testing.assert_array_equal(restored.coordinates, expected_coordinates)
    np.testing.assert_array_equal(restored.clustering["labels"], expected_labels)
    assert info["view"] == view and info["filters"] == A_FILTER
    assert info["selected_ids"] == ["b", "猫"] and info["primary"] == "猫"
    assert info["normalized"] is True
    assert info["summaries"] == workspace.info()["summaries"]
    assert restored.geometry == workspace.geometry


def test_mismatched_source_fingerprint_restore_preserves_current_session(workspace, source):
    archive = workspace.save({"color": "group"})
    write_dataset(source, IDS, VECTORS + 1, workspace.dataset.manifest,
                  names=NAMES, metadata=METADATA, overwrite=True)
    workspace.load(source)
    workspace.set_filters(B_FILTER)
    workspace.select(["007"], primary="007")
    before = workspace.info()
    matrix = workspace.matrix.copy()
    with pytest.raises(ValueError, match="Данные изменились"):
        workspace.restore(archive)
    assert workspace.info() == before
    np.testing.assert_array_equal(workspace.matrix, matrix)


def test_malformed_study_coordinates_rejected_before_replacing_working_session(workspace):
    compute(workspace)
    saved = read_study_bytes(workspace.save({}))
    saved["arrays"]["coordinates"] = saved["arrays"]["coordinates"][:-1]
    malformed = study_bytes(saved["state"], saved["arrays"])
    workspace.select(["007"], primary="007")
    before = workspace.info()
    coordinates = workspace.coordinates.copy()
    with pytest.raises(ValueError, match="Некорректные координаты"):
        workspace.restore(malformed)
    assert workspace.info() == before
    np.testing.assert_array_equal(workspace.coordinates, coordinates)


def test_exports_selected_order_and_uncomputed_coordinates_and_clusters_as_null(workspace):
    workspace.set_filters(A_FILTER)
    compute(workspace, scope="slice")
    compute(workspace, "clustering", "kmeans", {"n_clusters": 2}, scope="slice")
    workspace.select(["007", "猫", "b"], primary="b")
    records = pq.read_table(io.BytesIO(workspace.export("selected", "parquet"))).to_pylist()
    assert [row["id"] for row in records] == ["007", "猫", "b"]
    assert records[0]["coord0"] is None and records[0]["coord1"] is None and records[0]["cluster"] is None
    assert records[1]["coord0"] is not None and records[1]["cluster"] is not None
    csv_records = list(csv.DictReader(io.StringIO(workspace.export("all", "csv").decode())))
    assert [row["id"] for row in csv_records] == IDS
    assert csv_records[1]["coord0"] == "" and csv_records[1]["cluster"] == ""
    workspace.select([])
    assert pq.read_table(io.BytesIO(workspace.export("selected", "parquet"))).num_rows == 0


def test_map_sampling_is_stable_bounded_and_includes_requested_visible_object(tmp_path):
    count = 50_010
    ids = [f"row-{index}" for index in range(count)]
    vectors = np.column_stack([np.arange(count, dtype=np.float32), np.ones(count, dtype=np.float32)])
    source = write_dataset(tmp_path / "large", ids, vectors,
                           DatasetManifest("Large", "test", "v1", 2),
                           metadata=[{"visible": index % 2 == 0} for index in range(count)])
    session = Workspace(tmp_path / "cache")
    session.load(source)
    baseline = session.map_data()
    assert baseline["shown"] == 50_000 and baseline["available"] == count
    assert session.map_data()["ids"] == baseline["ids"]
    displayed_ids = set(baseline["ids"])
    absent = next(identifier for identifier in ids if identifier not in displayed_ids)
    included = session.map_data(include_id=absent)
    assert included["shown"] == 50_001
    assert included["ids"][:-1] == baseline["ids"] and included["ids"][-1] == absent
    session.set_filters([{"field": "visible", "op": "in", "values": [True]}])
    filtered = session.map_data(include_id="row-1")
    assert "row-1" not in filtered["ids"] and filtered["shown"] == count // 2
    assert filtered["geometry"] == baseline["geometry"]


def test_invalid_selection_request_does_not_partially_change_selection(workspace):
    workspace.select(["b"], primary="b")
    before = workspace.info()
    with pytest.raises(ValueError, match="Неизвестный ID"):
        workspace.select(["007"], primary="not-present")
    assert workspace.info() == before


def test_malformed_study_result_descriptor_rejected_without_corrupting_session(workspace):
    compute(workspace)
    saved = read_study_bytes(workspace.save({}))
    saved["state"]["projection"].pop("key")
    malformed = study_bytes(saved["state"], saved["arrays"])
    workspace.select(["007"], primary="007")
    before = workspace.info()
    with pytest.raises(ValueError):
        workspace.restore(malformed)
    assert workspace.info() == before


def test_cache_write_failure_keeps_computed_projection_and_clusters(workspace, monkeypatch):
    def failed_save(*args, **kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr(workspace.cache, "save", failed_save)
    assert compute(workspace)["status"] == "done"
    assert np.isfinite(workspace.coordinates).all()
    assert any("simulated disk full" in warning for warning in workspace.projection["warnings"])
    assert compute(workspace, "clustering", "kmeans", {"n_clusters": 2})["status"] == "done"
    assert len(workspace.clustering["labels"]) == 6
    assert sum(item["size"] for item in workspace.summaries) == 6
    assert any("simulated disk full" in warning for warning in workspace.clustering["warnings"])
