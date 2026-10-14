"""Search results behave like a single atomic clustering, including cancellation."""

import threading
import time

import numpy as np
import pytest

from embedding_visualizer.data import DatasetManifest, write_dataset
from embedding_visualizer.persistence import read_study_bytes, study_bytes
from embedding_visualizer.workspace import Workspace


@pytest.fixture
def session(tmp_path):
    rng = np.random.default_rng(77)
    x = np.vstack([rng.normal([v, v, 0], .3, (12, 3)) for v in (-4, 0, 4)]).astype(np.float32)
    source = write_dataset(tmp_path / "data", [str(i) for i in range(len(x))], x,
        DatasetManifest("Tuning", "test", "v1", 3), metadata=[{"keep": i % 3 != 0} for i in range(len(x))])
    result = Workspace(tmp_path / "cache")
    result.load(source)
    return result


def finish(session):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with session.lock:
            job = session.job.copy()
        if job["status"] != "running":
            return job
        time.sleep(.005)
    pytest.fail("Operation did not finish")


def search(session, **kwargs):
    session.start_operation("clustering", "kmeans",
        params={"n_clusters": 3, "space": "pca", "pca_components": 2, "random_state": 7},
        tuning={"objective": "inertia", "budget": 3}, **kwargs)
    job = finish(session)
    assert job["status"] == "done", job
    return job


def test_search_cache_and_study_restore_exact_winner_and_report(session, tmp_path, monkeypatch):
    assert search(session)["cache_hit"] is False
    info = session.info()["clustering"]
    labels = session.clustering["labels"].copy()
    assert info["tuning"]["completed"] == 3
    assert info["tuning"]["best_score"] == min(x["score"] for x in info["tuning"]["trials"])
    assert info["request"] == info["tuning"]["trials"][info["tuning"]["best_trial"] - 1]["params"]
    archive = session.save({"controls": {"cluster-mode": "auto"}, "x_range": [-2, 3]})

    def forbidden(*args, **kwargs):
        pytest.fail("Saved winner must not be recomputed")

    monkeypatch.setattr("embedding_visualizer.workspace.tune_clustering", forbidden)
    assert search(session)["cache_hit"] is True
    assert session.info()["clustering"]["tuning"] == info["tuning"]
    restored = Workspace(tmp_path / "fresh-cache")
    restored.restore(archive)
    np.testing.assert_array_equal(restored.clustering["labels"], labels)
    assert restored.clustering["tuning"] == info["tuning"]
    assert restored.info()["view"]["x_range"] == [-2, 3]


def test_auto_slice_labels_remain_aligned_and_map_geometry_unchanged(session):
    session.set_filters([{"field": "keep", "op": "in", "values": [True]}])
    geometry = session.geometry
    search(session, scope="slice")
    assert session.geometry == geometry
    assert np.all(session.clustering["labels"][~session.mask] == -2)
    assert len(np.unique(session.clustering["labels"][session.mask])) == 3
    assert session.clustering["tuning"]["sample_count"] == int(session.mask.sum())


def test_cancellation_after_search_before_publication_preserves_previous_result(session, monkeypatch):
    session.start_operation("clustering", "kmeans", {"n_clusters": 2})
    assert finish(session)["status"] == "done"
    old = session.clustering
    entered, release = threading.Event(), threading.Event()
    import embedding_visualizer.workspace as module
    summarize = module.summarize_clusters

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return summarize(*args, **kwargs)

    monkeypatch.setattr(module, "summarize_clusters", blocked)
    session.start_operation("clustering", "kmeans", {"n_clusters": 3}, tuning={"budget": 2})
    try:
        assert entered.wait(5)
        requested = session.cancel_job()
        assert requested["status"] == "running" and requested["cancel_requested"]
        assert session.cancel_job()["cancel_requested"]  # idempotent while the fit exits
        with pytest.raises(ValueError, match="Дождитесь"):
            session.start_operation("clustering", "kmeans", {"n_clusters": 2})
    finally:
        release.set()
    assert finish(session)["status"] == "cancelled"
    assert session.clustering is old


@pytest.mark.parametrize("method,params,tuning", [
    ("dbscan", {"n_clusters": 3}, {}),
    ("hdbscan", {"n_clusters": 3}, {}),
    ("kmeans", {"n_clusters": 3}, {"budget": True}),
    ("kmeans", {"n_clusters": 3}, {"budget": 1000}),
    ("kmeans", {"n_clusters": 3}, {"objective": "anything"}),
    ("kmeans", {"n_clusters": 3, "n_init": 10}, {}),
])
def test_invalid_search_rejected_before_replacing_job(session, method, params, tuning):
    with pytest.raises(ValueError):
        session.start_operation("clustering", method, params, tuning=tuning)
    assert session.job is None


def test_failed_search_keeps_previous_labels(session):
    search(session)
    old = session.clustering
    session.start_operation("clustering", "kmeans", {"n_clusters": 1},
                            tuning={"objective": "silhouette", "budget": 2})
    assert finish(session)["status"] == "error"
    assert session.clustering is old


def test_corrupt_search_winner_is_not_restored(session, tmp_path):
    search(session)
    decoded = read_study_bytes(session.save({}))
    decoded["state"]["clustering"]["tuning"]["best_trial"] = 999
    broken = study_bytes(decoded["state"], decoded["arrays"])
    fresh = Workspace(tmp_path / "another-cache")
    with pytest.raises(ValueError, match="best_trial"):
        fresh.restore(broken)


def test_manual_run_does_not_carry_previous_search_metadata(session):
    search(session)
    session.start_operation("clustering", "kmeans", {"n_clusters": 3})
    assert finish(session)["status"] == "done"
    assert "tuning" not in session.clustering


@pytest.mark.parametrize("change", ["request", "space", "labels"])
def test_inconsistent_search_study_is_rejected(session, tmp_path, change):
    search(session)
    decoded = read_study_bytes(session.save({}))
    descriptor = decoded["state"]["clustering"]
    if change == "request":
        descriptor["request"]["n_clusters"] = 7
    elif change == "space":
        descriptor["space"] = "original"
    else:
        decoded["arrays"]["labels"][0] = -1
    broken = study_bytes(decoded["state"], decoded["arrays"])
    with pytest.raises(ValueError, match="(Параметры|Метки)"):
        Workspace(tmp_path / "other").restore(broken)
