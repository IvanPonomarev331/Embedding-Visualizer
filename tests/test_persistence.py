"""Content identity, durable study round trips, and safe inspection exports."""

from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import io
import json
import zipfile

import numpy as np
import pyarrow.parquet as pq
import pytest

from embedding_visualizer.data.models import Dataset, DatasetManifest, ValidationReport
from embedding_visualizer.persistence import (
    ComputeCache,
    PersistenceError,
    dataset_fingerprint,
    export_rows,
    load_study,
    read_study_bytes,
    save_study,
    study_bytes,
)


@pytest.fixture
def dataset():
    return Dataset(
        ids=["0007", "b", "юзер"], names=["quoted, name", None, "Пример\nстроки"],
        embeddings=np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        metadata=[{"x": 1, "nested": {"b": True, "a": [1, None]}}, {}, {"group": "Тест"}],
        manifest=DatasetManifest("test", "space", "v1", 2),
        report=ValidationReport(3, 2),
    )


def test_fingerprint_deterministic_canonical_and_sensitive_to_every_source_field(dataset):
    original = dataset_fingerprint(dataset)
    same = copy.deepcopy(dataset)
    same.metadata[0] = {"nested": {"a": [1, None], "b": True}, "x": 1}
    assert dataset_fingerprint(same) == original
    changes = []
    for mutate in (
        lambda item: item.ids.__setitem__(0, "7"),
        lambda item: item.names.__setitem__(0, "renamed"),
        lambda item: item.metadata[0].__setitem__("x", 2),
        lambda item: item.embeddings.__setitem__((0, 0), 9),
        lambda item: setattr(item, "manifest", DatasetManifest("test", "space", "v2", 2)),
        lambda item: item.ids.reverse(),
    ):
        changed = copy.deepcopy(dataset)
        mutate(changed)
        changes.append(dataset_fingerprint(changed))
    assert original not in changes
    assert len(set(changes)) == len(changes)


def test_cache_key_canonical_and_invalidates_dataset_parameters_slice_and_operation(tmp_path):
    cache = ComputeCache(tmp_path)
    key = cache.key("fingerprint", "pca", {"seed": 42, "normalize": False}, [1, 0])
    assert key == cache.key("fingerprint", "pca", {"normalize": False, "seed": np.int64(42)}, np.array([1, 0]))
    alternatives = [
        cache.key("changed", "pca", {"seed": 42, "normalize": False}, [1, 0]),
        cache.key("fingerprint", "tsne", {"seed": 42, "normalize": False}, [1, 0]),
        cache.key("fingerprint", "pca", {"seed": 41, "normalize": False}, [1, 0]),
        cache.key("fingerprint", "pca", {"seed": 42, "normalize": True}, [1, 0]),
        cache.key("fingerprint", "pca", {"seed": 42, "normalize": False}, [0, 1]),
    ]
    assert key not in alternatives
    assert len(set(alternatives)) == len(alternatives)


def test_cache_roundtrip_and_immutable_concurrent_publish(tmp_path):
    cache = ComputeCache(tmp_path)
    key = cache.key("dataset", "projection", {})
    assert cache.load(key) is None

    def write(index):
        cache.save(key, {"coordinates": np.full((4, 2), index)}, {"winner": index})

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(write, range(12)))
    result = cache.load(key)
    assert result is not None
    assert np.all(result["arrays"]["coordinates"] == result["metadata"]["winner"])
    previous = result["metadata"].copy()
    cache.save(key, {"coordinates": np.zeros((4, 2))}, {"winner": "changed"})
    assert cache.load(key)["metadata"] == previous
    assert len(list(tmp_path.rglob("*.npz"))) == 1
    assert not list(tmp_path.rglob("*.tmp"))


def test_corrupt_cache_is_a_miss_and_key_cannot_escape_directory(tmp_path):
    cache = ComputeCache(tmp_path)
    key = cache.key("dataset", "pca", {})
    cache.save(key, {"coordinates": np.ones((2, 2))}, {})
    archive = next(tmp_path.rglob("*.npz"))
    archive.write_bytes(b"truncated zip")
    assert cache.load(key) is None
    with pytest.raises(PersistenceError, match="Cache key"):
        cache.load("../outside")
    with pytest.raises(PersistenceError, match="Cache key"):
        cache.save("../outside", {}, {})


def test_study_roundtrip_preserves_actual_results_settings_and_missing_slice_rows(tmp_path):
    state = {
        "fingerprint": "content-hash", "source": "/path/dataset.parquet",
        "selected_id": "0007", "filters": [{"field": "price", "operator": "gte", "value": 5}],
        "normalize": True, "projection": {"method": "umap", "random_state": 42},
        "view": {"x": 0, "y": 1}, "manifest": {"space_id": "model/layer"},
    }
    arrays = {"coordinates": np.array([[1.0, 2], [np.nan, np.nan], [3, 4]]),
              "labels": np.array([0, -2, -1], dtype=np.int64), "slice_indices": np.array([0, 2])}
    path = save_study(tmp_path / "study.npz", state, arrays)
    for result in (load_study(path), read_study_bytes(study_bytes(state, arrays))):
        assert result["state"] == state
        for name, array in arrays.items():
            np.testing.assert_array_equal(result["arrays"][name], array)
    save_study(path, {"replacement": True}, {})
    assert load_study(path) == {"state": {"replacement": True}, "arrays": {}}
    assert not list(tmp_path.glob("*.tmp"))


def test_rejects_pickle_object_arrays_and_invalid_json_without_overwriting(tmp_path):
    path = save_study(tmp_path / "study.npz", {}, {})
    original = path.read_bytes()
    with pytest.raises(PersistenceError, match="object arrays"):
        save_study(path, {}, {"unsafe": np.array([{}], dtype=object)})
    assert path.read_bytes() == original
    with pytest.raises(PersistenceError, match="NaN"):
        study_bytes({"value": float("nan")}, {})
    with pytest.raises(PersistenceError, match="infinity"):
        study_bytes({}, {"values": np.array([np.inf])})
    with pytest.raises(PersistenceError):
        read_study_bytes(b"not a zip file")


def _raw_archive(header, **arrays):
    output = io.BytesIO()
    np.savez_compressed(output, __header__=np.frombuffer(json.dumps(header).encode(), dtype=np.uint8), **arrays)
    return output.getvalue()


def test_rejects_unsupported_version_unexpected_members_and_loaded_object_arrays():
    header = {"format": "embedding-visualizer", "version": 999, "kind": "study", "state": {}, "arrays": []}
    with pytest.raises(PersistenceError, match="version"):
        read_study_bytes(_raw_archive(header))
    header["version"] = 1
    header["arrays"] = ["unsafe"]
    with pytest.raises(PersistenceError, match="object arrays are forbidden"):
        read_study_bytes(_raw_archive(header, array_0=np.array([{}], dtype=object)))
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("../../outside.npy", b"payload")
    with pytest.raises(PersistenceError, match="unexpected files"):
        read_study_bytes(output.getvalue())


def test_expanded_archive_size_guard_runs_before_numpy_load(monkeypatch):
    from embedding_visualizer import persistence

    data = study_bytes({}, {"zeros": np.zeros((1000, 2))})
    monkeypatch.setattr(persistence, "_MAX_ARCHIVE_BYTES", 2000)
    assert len(data) < 2000
    with pytest.raises(PersistenceError, match="Expanded archive"):
        read_study_bytes(data)


@pytest.mark.parametrize("format", ["csv", "parquet"])
def test_export_preserves_ids_order_metadata_results_and_missing_values(dataset, format):
    coords = np.array([[1.5, 2], [np.nan, np.nan], [3, 4]])
    labels = np.array([0, -2, -1])
    data = export_rows(dataset, [2, 0, 1], coords, labels, format)
    if format == "parquet":
        records = pq.read_table(io.BytesIO(data)).to_pylist()
    else:
        records = list(csv.DictReader(io.StringIO(data.decode())))
    assert [item["id"] for item in records] == ["юзер", "0007", "b"]
    assert records[0]["name"] == "Пример\nстроки"
    assert json.loads(records[1]["metadata"]) == dataset.metadata[0]
    assert float(records[0]["coord0"]) == 3
    assert int(records[0]["cluster"]) == -1
    assert records[2]["coord0"] in (None, "")
    assert records[2]["cluster"] in (None, "")


@pytest.mark.parametrize("format", ["csv", "parquet"])
def test_empty_export_keeps_schema(dataset, format):
    data = export_rows(dataset, [], np.ones((3, 2)), np.array([1, -1, -2]), format)
    if format == "parquet":
        table = pq.read_table(io.BytesIO(data))
        assert table.num_rows == 0
        assert table.column_names == ["id", "name", "metadata", "coord0", "coord1", "cluster"]
    else:
        assert data.decode() == "id,name,metadata,coord0,coord1,cluster\n"


@pytest.mark.parametrize("indices,coordinates,labels,error", [
    ([3], None, None, "outside"),
    ([-1], None, None, "outside"),
    ([1.1], None, None, "integer sequence"),
    ([True, 1], None, None, "integer sequence"),
    ([1], np.ones((1, 2)), None, "Coordinates"),
    ([1], None, np.array([1]), "Labels"),
])
def test_export_rejects_misaligned_results_and_invalid_rows(dataset, indices, coordinates, labels, error):
    with pytest.raises(PersistenceError, match=error):
        export_rows(dataset, indices, coordinates, labels)


def test_tiny_archive_cannot_claim_a_huge_array_allocation():
    raw = io.BytesIO()
    np.lib.format.write_array_header_1_0(raw, {"descr": "<f8", "fortran_order": False, "shape": (10**12,)})
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("array_0.npy", raw.getvalue())
    with pytest.raises(PersistenceError, match="Array shape"):
        read_study_bytes(output.getvalue())
