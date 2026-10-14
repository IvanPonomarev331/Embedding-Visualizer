"""Exercise real HTTP boundaries, downloads and independent browser sessions."""

import http.client
import json
from pathlib import Path
import threading
import time

import pytest

from embedding_visualizer.data import DatasetManifest, write_dataset
from embedding_visualizer.settings import Settings
from embedding_visualizer.web.server import LocalServer


@pytest.fixture
def server(tmp_path):
    path = write_dataset(tmp_path / "data" / "example", ["a", "b", "c"],
        [[1, 0], [.9, .1], [0, 1]], DatasetManifest("HTTP example", "s", "v", 2),
        names=["Alpha", "Beta", "Gamma"], metadata=[{"type": "x"}, {"type": "x"}, {"type": "y"}])
    server = LocalServer(Settings(tmp_path / "data", tmp_path / "cache", port=0))
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .02}, daemon=True)
    thread.start()
    yield server, path
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def request(server, path, data=None, *, raw=None, workspace=None, auth=True, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    merged = {"Content-Type": "application/json"}
    if auth:
        merged["X-Visualizer-Token"] = server.application.token
    if workspace:
        merged["X-Visualizer-Workspace"] = workspace
    merged.update(headers or {})
    body = raw if raw is not None else (json.dumps(data).encode() if data is not None else None)
    connection.request("POST" if body is not None else "GET", path, body=body, headers=merged)
    response = connection.getresponse()
    content = response.read()
    status, response_headers = response.status, dict(response.getheaders())
    connection.close()
    if response_headers.get("Content-Type", "").startswith("application/json"):
        content = json.loads(content)
    return status, content, response_headers


def session(server):
    status, result, _ = request(server, "/api/workspace", {})
    assert status == 200
    return result["workspace"]


def test_local_entry_assets_and_config(server):
    server, path = server
    status, content, headers = request(server, "/", auth=False)
    assert status == 200 and b"Embedding Visualizer" in content
    assert headers["Content-Type"].startswith("text/html")
    status, content, _ = request(server, "/api/config", auth=False)
    assert status == 200
    assert content["default_path"] == str(path)
    assert content["datasets"] == [{"path": str(path), "name": "HTTP example"}]
    assert content["token"] == server.application.token
    assert request(server, "/assets/../workspace.py")[0] == 404


def test_cross_origin_host_and_auth_are_rejected(server):
    server, _ = server
    assert request(server, "/api/workspace", {}, auth=False)[0] == 403
    assert request(server, "/api/config", headers={"Origin": "https://unrelated.example"})[0] == 403
    assert request(server, "/api/config", headers={"Host": "unrelated.example"})[0] == 403
    assert request(server, "/api/workspace", {}, headers={"Content-Type": "text/plain"})[0] == 400


def test_browser_sessions_are_independent_and_failed_load_preserves_data(server):
    server, path = server
    a, b = session(server), session(server)
    assert request(server, "/api/load", {"path": str(path)}, workspace=a)[0] == 200
    assert request(server, "/api/state", workspace=b)[0] == 400
    status, points, _ = request(server, "/api/map", {"x": 0, "y": 1}, workspace=a)
    assert status == 200 and points["ids"] == ["a", "b", "c"]
    assert request(server, "/api/load", {"path": str(path / "missing")}, workspace=a)[0] == 400
    status, result, _ = request(server, "/api/state", workspace=a)
    assert status == 200 and result["count"] == 3


def test_http_filter_neighbors_and_export_agree_on_ids(server):
    server, path = server
    workspace = session(server)
    request(server, "/api/load", {"path": str(path)}, workspace=workspace)
    status, result, _ = request(server, "/api/filters", {"filters": [{"field": "type", "op": "in", "values": ["x"]}]}, workspace=workspace)
    assert status == 200 and result["visible_count"] == 2
    status, result, _ = request(server, "/api/object?id=c&scope=slice&k=2&metric=euclidean", workspace=workspace)
    assert status == 200 and not result["in_slice"]
    assert [row["id"] for row in result["neighbors"]] == ["b", "a"]
    status, content, headers = request(server, "/api/export", {"scope": "slice", "format": "csv"}, workspace=workspace)
    assert status == 200
    assert len(content.decode().splitlines()) == 3
    assert "embeddings.csv" in headers["Content-Disposition"]


def test_study_download_upload_restores_exact_projection_and_view(server):
    server, path = server
    workspace = session(server)
    request(server, "/api/load", {"path": str(path)}, workspace=workspace)
    status, result, _ = request(server, "/api/operation", {"operation": "projection", "method": "pca", "params": {"n_components": 2}}, workspace=workspace)
    assert status == 200
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        _, job, _ = request(server, "/api/job", workspace=workspace)
        if job["status"] != "running":
            break
        time.sleep(.02)
    assert job["status"] == "done"
    _, before, _ = request(server, "/api/map", {"x": 0, "y": 1}, workspace=workspace)
    view = {"axes": [0, 1], "viewport": {"xaxis.range": [-1, .25], "yaxis.range": [-.1, .1]}}
    status, archive, _ = request(server, "/api/study/save", {"view": view}, workspace=workspace)
    assert status == 200 and archive.startswith(b"PK")
    other = session(server)
    status, restored, _ = request(server, "/api/study/open", raw=archive, workspace=other,
                                   headers={"Content-Type": "application/octet-stream"})
    assert status == 200 and restored["view"] == view
    _, after, _ = request(server, "/api/map", {"x": 0, "y": 1}, workspace=other)
    assert before == after


def test_auto_search_http_progress_and_cancel_are_session_scoped(server, monkeypatch):
    server, path = server
    identifier, other = session(server), session(server)
    request(server, "/api/load", {"path": str(path)}, workspace=identifier)
    entered, release = threading.Event(), threading.Event()
    import embedding_visualizer.workspace as module
    tune = module.tune_clustering

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return tune(*args, **kwargs)

    monkeypatch.setattr(module, "tune_clustering", blocked)
    status, job, _ = request(server, "/api/operation", {
        "operation": "clustering", "method": "kmeans", "params": {"n_clusters": 2},
        "tuning": {"objective": "inertia", "budget": 2}}, workspace=identifier)
    assert status == 200 and job["cancellable"]
    try:
        assert entered.wait(5)
        assert request(server, "/api/job/cancel", {}, workspace=identifier, auth=False)[0] == 403
        assert request(server, "/api/job/cancel", {}, workspace=other)[1]["status"] == "idle"
        status, cancelled, _ = request(server, "/api/job/cancel", {}, workspace=identifier)
        assert status == 200 and cancelled["cancel_requested"]
        assert cancelled["status"] == "running"
    finally:
        release.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        _, job, _ = request(server, "/api/job", workspace=identifier)
        if job["status"] != "running":
            break
        time.sleep(.005)
    assert job["status"] == "cancelled"
    assert request(server, "/api/state", workspace=identifier)[1]["clustering"] is None
