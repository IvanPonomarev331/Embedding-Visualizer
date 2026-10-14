"""Automatic clustering scores, repeatability, bounded search and cancellation."""

from dataclasses import replace
import json

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

from embedding_visualizer.compute import tuning
from embedding_visualizer.compute.clustering import ClusteringResult, cluster
from embedding_visualizer.compute.common import ComputeError
from embedding_visualizer.compute.tuning import TUNABLE_METHODS, TuningCancelled, tune_clustering


@pytest.fixture
def blobs():
    return make_blobs(n_samples=60, centers=3, n_features=4, cluster_std=0.4, random_state=7)[0]


def empirical_sse(matrix, labels):
    return sum(np.sum((matrix[labels == label] - matrix[labels == label].mean(axis=0)) ** 2)
               for label in np.unique(labels))


@pytest.mark.parametrize("method", TUNABLE_METHODS)
@pytest.mark.parametrize("objective", ["inertia", "silhouette"])
def test_scores_and_best_are_valid_and_no_worse_than_baseline(blobs, method, objective):
    before = blobs.copy()
    outcome = tune_clustering(blobs, method, n_clusters=3, objective=objective, budget=6)
    report = outcome.report
    successful = [trial for trial in report["trials"] if trial["status"] == "ok"]
    assert report["trials"][0]["status"] == "ok"
    assert len(np.unique(outcome.result.labels)) == 3
    assert min(outcome.result.labels) >= 0
    expected = (empirical_sse(blobs, outcome.result.labels) if objective == "inertia"
                else silhouette_score(blobs, outcome.result.labels, metric="euclidean"))
    assert report["best_score"] == pytest.approx(expected)
    best = min(trial["score"] for trial in successful) if objective == "inertia" else max(trial["score"] for trial in successful)
    assert report["best_score"] == best
    assert report["trials"][report["best_trial"] - 1]["score"] == best
    assert report["sample_count"] == len(blobs)
    assert report["score_metric"] == "euclidean"
    assert report["completed"] == report["total"] == 6
    json.dumps(report, allow_nan=False)
    manual = cluster(blobs, method, **outcome.request)
    np.testing.assert_array_equal(outcome.result.labels, manual.labels)
    np.testing.assert_array_equal(blobs, before)


def test_known_sse_is_a_sum_around_empirical_centroids():
    matrix = np.array([[0., 0.], [2, 0], [10, 0], [14, 0]])
    result = tune_clustering(matrix, "kmeans", n_clusters=2, budget=1)
    assert result.report["best_score"] == pytest.approx(10.0)


def test_inertia_handles_one_cluster_and_constant_zero_data():
    result = tune_clustering(np.zeros((8, 3)), "kmeans", n_clusters=1, budget=2)
    assert result.report["best_score"] == 0
    assert result.report["best_trial"] == 1


def test_stable_sse_does_not_overflow_constant_huge_columns_or_erase_small_columns():
    matrix = np.array([[1e308, 1.], [1e308, 3.], [1e308, 5.]])
    assert tuning._inertia(matrix, np.zeros(3, dtype=int)) == pytest.approx(8.0)
    assert tuning._inertia(np.full((5, 2), 1e308), np.zeros(5, dtype=int)) == 0


def test_reproducible_candidate_order_scores_and_labels(blobs):
    first = tune_clustering(blobs, "gaussian_mixture", n_clusters=3, budget=12, random_state=121)
    second = tune_clustering(blobs, "gaussian_mixture", n_clusters=3, budget=12, random_state=121)
    assert first.request == second.request
    assert first.report["best_trial"] == second.report["best_trial"]
    assert first.report["best_score"] == second.report["best_score"]
    assert [trial["params"] for trial in first.report["trials"]] == [trial["params"] for trial in second.report["trials"]]
    np.testing.assert_array_equal(first.result.labels, second.result.labels)


def test_pca_is_fitted_once_and_manual_winning_request_reproduces_it(monkeypatch, blobs):
    actual_project = tuning.project
    projections = []
    representations = []
    actual_cluster = tuning.cluster

    def counted_project(matrix, method, **params):
        projections.append(params)
        return actual_project(matrix, method, **params)

    def observed_cluster(matrix, method, **params):
        representations.append(matrix.copy())
        result = actual_cluster(matrix, method, **params)
        # Force the winning clustering seed to differ from the PCA seed.
        return replace(result, labels=np.zeros(len(matrix), dtype=int)) if len(representations) == 1 else result

    monkeypatch.setattr(tuning, "project", counted_project)
    monkeypatch.setattr(tuning, "cluster", observed_cluster)
    result = tune_clustering(blobs, "kmeans", n_clusters=3, budget=4, space="pca", pca_components=2, random_state=17)
    assert len(projections) == 1
    assert projections[0]["random_state"] == 17
    for representation in representations:
        np.testing.assert_array_equal(representation, result.result.representation)
    assert result.result.space == result.report["score_space"] == "pca"
    assert result.report["representation_seed"] == result.report["base_seed"] == 17
    assert result.request["random_state"] != 17
    assert result.report["representation_parameters"]["svd_solver"] == "full"
    assert result.result.parameters["pca_components"] == result.request["pca_components"] == 2
    manual = cluster(blobs, "kmeans", **result.request)
    np.testing.assert_array_equal(manual.labels, result.result.labels)
    np.testing.assert_array_equal(manual.representation, result.result.representation)


def test_silhouette_uses_one_shared_bounded_sample_but_fits_every_row(monkeypatch):
    matrix = np.random.default_rng(12).normal(size=(1500, 3))
    observed_samples = []
    fitted_sizes = []

    def fake_cluster(data, method, **params):
        fitted_sizes.append(len(data))
        labels = np.arange(len(data), dtype=int) % 2
        return ClusteringResult(labels, method, params.copy(), "original", data)

    def fake_silhouette(data, labels, *, metric):
        observed_samples.append(data.copy())
        assert metric == "euclidean"
        return 0.25

    monkeypatch.setattr(tuning, "cluster", fake_cluster)
    monkeypatch.setattr(tuning, "silhouette_score", fake_silhouette)
    first = tune_clustering(matrix, "kmeans", n_clusters=2, budget=3, objective="silhouette")
    second = tune_clustering(matrix, "kmeans", n_clusters=2, budget=1, objective="silhouette")
    assert fitted_sizes == [1500] * 4
    assert first.report["sample_count"] == second.report["sample_count"] == 1000
    assert len(observed_samples) == 4
    for sampled in observed_samples:
        assert len(sampled) == 1000
        np.testing.assert_array_equal(sampled, observed_samples[0])


def test_failed_nonfinite_and_wrong_count_candidates_are_skipped(monkeypatch, blobs):
    calls = []

    def sometimes_fails(matrix, method, **params):
        calls.append(params)
        if len(calls) == 1:
            raise ComputeError("synthetic fit failure")
        result = cluster(matrix, method, **params)
        if len(calls) == 2:
            return replace(result, labels=np.zeros(len(matrix), dtype=int))
        if len(calls) == 3:
            return replace(result, labels=np.full(len(matrix), np.nan))
        return result

    monkeypatch.setattr(tuning, "cluster", sometimes_fails)
    result = tune_clustering(blobs, "kmeans", n_clusters=3, budget=4)
    assert [trial["status"] for trial in result.report["trials"]] == ["error", "error", "error", "ok"]
    assert result.report["trials"][1]["actual_clusters"] == 1
    assert result.report["best_trial"] == 4
    json.dumps(result.report, allow_nan=False)


def test_noise_is_not_a_valid_fixed_k_result(monkeypatch, blobs):
    def noisy(matrix, method, **params):
        result = cluster(matrix, method, **params)
        result.labels[0] = -1
        return result

    monkeypatch.setattr(tuning, "cluster", noisy)
    with pytest.raises(ComputeError, match="no noise") as caught:
        tune_clustering(blobs, "kmeans", n_clusters=3, budget=2)
    assert caught.value.report["completed"] == 2
    assert caught.value.report["best_score"] is None


def test_nonfinite_scores_are_failed_trials(monkeypatch, blobs):
    monkeypatch.setattr(tuning, "_inertia", lambda *args: float("inf"))
    with pytest.raises(ComputeError, match="non-finite objective") as caught:
        tune_clustering(blobs, "kmeans", n_clusters=3, budget=1)
    assert caught.value.report["trials"][0]["score"] is None
    json.dumps(caught.value.report, allow_nan=False)


def test_duplicate_vectors_cannot_silently_reduce_requested_k():
    with pytest.raises(ComputeError, match="exactly 2 clusters") as caught:
        tune_clustering(np.zeros((6, 3)), "kmeans", n_clusters=2, budget=2)
    assert all(trial["actual_clusters"] == 1 for trial in caught.value.report["trials"])


@pytest.mark.parametrize("method", TUNABLE_METHODS)
def test_candidate_budget_and_deduplication(method):
    candidates = tuning._candidates(method, 3, 80, 2**32 - 1, 48)
    assert 1 <= len(candidates) <= 48
    assert len({json.dumps(candidate, sort_keys=True) for candidate in candidates}) == len(candidates)
    assert candidates[0]["random_state"] == 2**32 - 1
    assert all(0 <= candidate["random_state"] < 2**32 for candidate in candidates)
    if method == "agglomerative":
        assert len(candidates) == 7
    if method == "spectral":
        assert candidates[0]["n_neighbors"] == 15
        assert all(2 <= candidate["n_neighbors"] <= 80 for candidate in candidates)


def test_agglomerative_skips_invalid_cosine_candidates_without_losing_baseline():
    result = tune_clustering([[0, 0], [1, 1], [10, 10], [12, 11]], "agglomerative", n_clusters=2, budget=48)
    assert result.report["total"] == result.report["completed"] == 7
    assert [trial["status"] for trial in result.report["trials"]] == ["ok"] * 4 + ["error"] * 3
    assert result.report["best_score"] is not None


@pytest.mark.parametrize("params,error", [
    ({"budget": 0}, "budget"),
    ({"budget": 49}, "budget"),
    ({"budget": True}, "budget"),
    ({"budget": 2.5}, "budget"),
    ({"n_clusters": True}, "n_clusters"),
    ({"n_clusters": 0}, "n_clusters"),
    ({"n_clusters": 7}, "n_clusters"),
    ({"n_clusters": 2.1}, "n_clusters"),
    ({"random_state": True}, "random_state"),
    ({"random_state": 2**32}, "random_state"),
    ({"objective": "calinski"}, "objective"),
    ({"objective": "silhouette", "n_clusters": 1}, "Silhouette"),
    ({"objective": "silhouette", "n_clusters": 6}, "Silhouette"),
    ({"space": "umap"}, "space"),
    ({"space": "pca"}, "pca_components"),
    ({"space": "pca", "pca_components": True}, "pca_components"),
    ({"space": "pca", "pca_components": 4}, "pca_components"),
    ({"pca_components": 2}, "only valid"),
    ({"progress": True}, "progress"),
    ({"cancelled": False}, "cancelled"),
])
def test_invalid_untrusted_parameters_rejected_before_fit(monkeypatch, params, error):
    monkeypatch.setattr(tuning, "cluster", lambda *args, **kwargs: pytest.fail("Must validate before fitting"))
    request = {"n_clusters": 2, **params}
    with pytest.raises(ComputeError, match=error):
        tune_clustering(np.ones((6, 3)), "kmeans", **request)


@pytest.mark.parametrize("method,error", [("dbscan", "manual"), ("hdbscan", "manual"), ("other", "method")])
def test_methods_without_exact_k_are_not_searched(method, error):
    with pytest.raises(ComputeError, match=error):
        tune_clustering(np.ones((6, 3)), method, n_clusters=2)


def test_progress_includes_initial_state_and_every_trial_and_is_observer_safe(blobs):
    snapshots = []

    def observe(snapshot):
        snapshots.append(snapshot)
        if snapshot["trial"]:
            snapshot["trial"]["params"]["n_clusters"] = 999

    result = tune_clustering(blobs, "kmeans", n_clusters=3, budget=3, progress=observe)
    assert [snapshot["completed"] for snapshot in snapshots] == [0, 1, 2, 3]
    assert all(snapshot["total"] == 3 for snapshot in snapshots)
    assert snapshots[0]["best_score"] is None
    assert snapshots[0]["trial"] is None
    assert result.request["n_clusters"] == 3
    assert all(trial["params"]["n_clusters"] == 3 for trial in result.report["trials"])


def test_cancellation_before_fit_or_pca_publishes_nothing(monkeypatch, blobs):
    monkeypatch.setattr(tuning, "cluster", lambda *args, **kwargs: pytest.fail("Cancelled before fit"))
    monkeypatch.setattr(tuning, "project", lambda *args, **kwargs: pytest.fail("Cancelled before PCA"))
    with pytest.raises(TuningCancelled):
        tune_clustering(blobs, "kmeans", n_clusters=3, space="pca", pca_components=2, cancelled=lambda: True)


def test_cancellation_after_fit_does_not_score_or_publish_partial_best(monkeypatch, blobs):
    state = {"cancelled": False, "fits": 0}

    def fitted(matrix, method, **params):
        result = cluster(matrix, method, **params)
        state["fits"] += 1
        state["cancelled"] = True
        return result

    monkeypatch.setattr(tuning, "cluster", fitted)
    monkeypatch.setattr(tuning, "_inertia", lambda *args: pytest.fail("Cancelled before scoring"))
    with pytest.raises(TuningCancelled):
        tune_clustering(blobs, "kmeans", n_clusters=3, budget=3, cancelled=lambda: state["cancelled"])
    assert state["fits"] == 1


def test_cancellation_after_a_successful_trial_still_returns_no_result(blobs):
    state = {"cancelled": False}

    def progress(snapshot):
        if snapshot["completed"] == 1:
            state["cancelled"] = True

    with pytest.raises(TuningCancelled, match="no result was applied"):
        tune_clustering(blobs, "kmeans", n_clusters=3, budget=3,
                        progress=progress, cancelled=lambda: state["cancelled"])
