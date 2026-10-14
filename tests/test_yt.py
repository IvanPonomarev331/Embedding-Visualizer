from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import traceback
from types import SimpleNamespace

import pytest

from embedding_visualizer.adapters import yt
from embedding_visualizer.data import DatasetValidationError, load_dataset


@dataclass
class FakeTablePath:
    name: str
    columns: list[str]
    start_index: int
    end_index: int


class FakeStream:
    def __init__(self, rows):
        self.rows = iter(rows)
        self.closed = False
        self.read_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.read_count += 1
        return next(self.rows)

    def close(self):
        self.closed = True


class FakeClient:
    TablePath = FakeTablePath

    def __init__(self, rows, *, dynamic=False, node_type="table"):
        self.rows = rows
        self.dynamic = dynamic
        self.node_type = node_type
        self.get_calls = []
        self.read_calls = []
        self.stream = None

    def get(self, path):
        self.get_calls.append(path)
        return self.dynamic if path.endswith("/@dynamic") else self.node_type

    def read_table(self, path, **kwargs):
        self.read_calls.append((path, kwargs))
        self.stream = FakeStream(self.rows[path.start_index:path.end_index])
        return self.stream


def run_import(tmp_path, rows=None, *, client=None, **kwargs):
    defaults = dict(
        columns={"id": "key", "embedding": "vector"},
        dataset_name="YT import",
        space_id="space-v1",
        version="snapshot-1",
    )
    defaults.update(kwargs)
    client = client if client is not None else FakeClient(rows or [])
    result = yt.import_yt_dataset(
        "test-cluster", "//home/test/vectors", tmp_path / "dataset", client=client, **defaults
    )
    return result, client


def test_bounded_remote_slice_and_mapping(tmp_path):
    rows = [
        {"key": 0, "vector": [1, 0]},
        {"key": "001", "vector": [0, 1], "title": "First", "attrs": {"group": "a"}, "size": 2},
        {"key": 9007199254740993, "vector": [1, 1], "title": None,
         "attrs": json.dumps({"nested": {"flag": True}}), "size": 2},
        {"key": 3, "vector": [2, 2]},
    ]
    result, client = run_import(
        tmp_path, rows, offset=1, limit=2,
        columns={"id": "key", "embedding": "vector", "name": "title", "metadata": "attrs", "embedding_size": "size"},
    )
    dataset = load_dataset(result)
    assert dataset.ids == ["001", "9007199254740993"]
    assert dataset.names == ["First", None]
    assert dataset.metadata == [{"group": "a"}, {"nested": {"flag": True}}]
    assert dataset.manifest.embedding_dim == 2
    path, options = client.read_calls[0]
    assert path.start_index == 1 and path.end_index == 3
    assert path.columns == ["key", "vector", "title", "attrs", "size"]
    assert options == {"format": "json", "raw": False, "unordered": False}
    assert client.stream.closed
    assert client.stream.read_count == 2


def test_no_sdk_needed_for_injected_client(tmp_path, monkeypatch):
    def unexpected_import(name):
        raise AssertionError("SDK must stay optional")

    monkeypatch.setattr(yt, "import_module", unexpected_import)
    result, _ = run_import(tmp_path, [{"key": "a", "vector": [1, 2]}])
    assert load_dataset(result).metadata == [{}]


def test_missing_sdk_is_actionable_and_does_not_echo_exception(tmp_path, monkeypatch):
    def missing(name):
        raise ImportError("do-not-print-this")

    monkeypatch.setattr(yt, "import_module", missing)
    with pytest.raises(DatasetValidationError, match="pip install ytsaurus-client") as caught:
        yt.import_yt_dataset(
            "cluster", "//table", tmp_path / "dataset", columns={"id": "id", "embedding": "embedding"},
            dataset_name="test", space_id="s", version="1",
        )
    assert "do-not-print-this" not in "".join(traceback.format_exception(caught.value))


def test_sdk_client_inherits_environment_config_without_mutating_global(monkeypatch):
    configuration = {"token": "never-log-this", "proxy": {"url": "original"}}
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return "configured-client"

    sdk = SimpleNamespace(
        YtClient=create,
        config=SimpleNamespace(get_config=lambda client: configuration),
    )
    monkeypatch.setattr(yt, "import_module", lambda name: sdk)
    assert yt._new_client("explicit-proxy") == "configured-client"
    assert captured["proxy"] == "explicit-proxy"
    assert captured["config"] == configuration
    assert captured["config"] is not configuration
    assert captured["config"]["proxy"] is not configuration["proxy"]


@pytest.mark.parametrize("overrides, message", [
    ({"limit": 0}, "positive integer"),
    ({"limit": True}, "positive integer"),
    ({"offset": -1}, "non-negative integer"),
    ({"offset": 0.5}, "non-negative integer"),
    ({"columns": {"id": "key"}}, "require id and embedding"),
    ({"columns": {"id": "key", "embedding": "vector", "extra": "x"}}, "require id and embedding"),
    ({"columns": {"id": "", "embedding": "vector"}}, "non-empty string"),
])
def test_invalid_request_does_not_read_remote(tmp_path, overrides, message):
    client = FakeClient([])
    with pytest.raises(DatasetValidationError, match=message):
        run_import(tmp_path, client=client, **overrides)
    assert not client.get_calls and not client.read_calls


@pytest.mark.parametrize("client, message", [
    (FakeClient([], dynamic=True), "static tables only"),
    (FakeClient([], node_type="map_node"), "must be a table"),
])
def test_rejects_unsupported_source_before_read(tmp_path, client, message):
    with pytest.raises(DatasetValidationError, match=message):
        run_import(tmp_path, client=client)
    assert not client.read_calls


def test_empty_slice_creates_no_dataset(tmp_path):
    with pytest.raises(DatasetValidationError, match="slice is empty"):
        run_import(tmp_path, [{"key": "a", "vector": [1]}], offset=5)
    assert not (tmp_path / "dataset").exists()


@pytest.mark.parametrize("rows, message", [
    ([{"key": "a"}], "columns are missing"),
    ([{"key": "a", "vector": []}], "non-empty numeric vector"),
    ([{"key": "a", "vector": [1, 2]}, {"key": "b", "vector": [1]}], "embedding length"),
    ([{"key": "a", "vector": [float("nan")]}], "must be finite"),
    ([{"key": 1, "vector": [1]}, {"key": "1", "vector": [2]}], "Duplicate YT ID"),
    ([{"key": 1.0, "vector": [1]}], "floating-point IDs"),
])
def test_invalid_schema_and_data_do_not_write(tmp_path, rows, message):
    with pytest.raises(DatasetValidationError, match=message):
        run_import(tmp_path, rows)
    assert not (tmp_path / "dataset").exists()


@pytest.mark.parametrize("size", [3, None, True, 2.0])
def test_embedding_size_must_match_vector(tmp_path, size):
    with pytest.raises(DatasetValidationError, match="embedding_size"):
        run_import(tmp_path, [{"key": "a", "vector": [1, 2], "size": size}],
                   columns={"id": "key", "embedding": "vector", "embedding_size": "size"})


@pytest.mark.parametrize("metadata", ["invalid-json", "[]", ["wrong"], {"bad": float("inf")}])
def test_invalid_metadata_rejected(tmp_path, metadata):
    with pytest.raises(DatasetValidationError, match="metadata"):
        run_import(tmp_path, [{"key": "a", "vector": [1], "attrs": metadata}],
                   columns={"id": "key", "embedding": "vector", "metadata": "attrs"})


def test_remote_failure_hides_sensitive_sdk_details(tmp_path):
    class FailingClient(FakeClient):
        def read_table(self, *args, **kwargs):
            raise RuntimeError("Authorization: secret-token")

    with pytest.raises(DatasetValidationError, match="read failed") as caught:
        run_import(tmp_path, client=FailingClient([]))
    assert "secret-token" not in "".join(traceback.format_exception(caught.value))
    assert not (tmp_path / "dataset").exists()


def test_closes_stream_after_iteration_failure(tmp_path):
    class FailingStream(FakeStream):
        def __next__(self):
            raise RuntimeError("private-sdk-detail")

    class FailingClient(FakeClient):
        def read_table(self, *args, **kwargs):
            self.stream = FailingStream([])
            return self.stream

    client = FailingClient([])
    with pytest.raises(DatasetValidationError, match="read failed"):
        run_import(tmp_path, client=client)
    assert client.stream.closed
