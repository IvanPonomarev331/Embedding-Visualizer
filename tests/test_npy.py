from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from embedding_visualizer.adapters import convert_npy
from embedding_visualizer.data import DatasetManifest, DatasetValidationError, load_dataset


@pytest.fixture
def inputs(tmp_path: Path):
    matrix = tmp_path / "vectors.npy"
    np.save(matrix, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps([{"id": "a", "name": "A"}, {"id": "b", "name": "B"}]))
    manifest = DatasetManifest(
        dataset_name="Import test",
        space_id="test-space",
        version="1",
        embedding_dim=2,
        default_metric="cosine",
    )
    return matrix, rows, tmp_path / "converted", manifest


def test_requires_explicit_row_identity(inputs):
    matrix, rows, output, manifest = inputs
    with pytest.raises(DatasetValidationError, match="row identity"):
        convert_npy(matrix, rows, output, manifest)
    assert not output.exists()


def test_acknowledged_order_roundtrip(inputs):
    matrix, rows, output, manifest = inputs
    assert convert_npy(matrix, rows, output, manifest, assume_row_order=True) == output
    dataset = load_dataset(output)
    assert dataset.ids == ["a", "b"]
    assert dataset.names == ["A", "B"]
    np.testing.assert_array_equal(dataset.embeddings, [[1, 2], [3, 4]])


@pytest.mark.parametrize("ids_format", ["json", "npy"])
def test_explicit_ids_reorder_metadata_only(inputs, ids_format):
    matrix, rows, output, manifest = inputs
    rows.write_text(json.dumps([
        {"id": "a", "name": "A", "metadata": {"nested": {"value": 7}}, "frequency": 10},
        {"id": "b", "name": "B", "metadata": {"nested": {"value": 8}}, "frequency": 20},
    ]))
    ids = rows.with_name(f"ids.{ids_format}")
    if ids_format == "json":
        ids.write_text('["b", "a"]')
    else:
        np.save(ids, np.array(["b", "a"]))
    convert_npy(matrix, rows, output, manifest, ids_path=ids)
    dataset = load_dataset(output)
    assert dataset.ids == ["b", "a"]
    assert dataset.names == ["B", "A"]
    assert dataset.metadata == [
        {"nested": {"value": 8}, "frequency": 20},
        {"nested": {"value": 7}, "frequency": 10},
    ]
    np.testing.assert_array_equal(dataset.embeddings, [[1, 2], [3, 4]])


def test_csv_preserves_leading_zero_ids(inputs):
    matrix, rows, output, manifest = inputs
    csv_path = rows.with_suffix(".csv")
    csv_path.write_text('id,name,metadata,group\n001,First,"{""enabled"":true}",x\n002,Second,,y\n')
    convert_npy(matrix, csv_path, output, manifest, assume_row_order=True)
    dataset = load_dataset(output)
    assert dataset.ids == ["001", "002"]
    assert dataset.metadata == [{"enabled": True, "group": "x"}, {"group": "y"}]


def test_parquet_metadata_and_custom_columns(inputs):
    matrix, rows, output, manifest = inputs
    parquet_path = rows.with_suffix(".parquet")
    pq.write_table(pa.Table.from_pylist([
        {"item_id": "a", "title": "A", "metadata": {"available": True}, "price": 7.5},
        {"item_id": "b", "title": "B", "metadata": {"available": False}, "price": 9.0},
    ]), parquet_path)
    convert_npy(matrix, parquet_path, output, manifest,
                id_column="item_id", name_column="title", assume_row_order=True)
    dataset = load_dataset(output)
    assert dataset.ids == ["a", "b"]
    assert dataset.metadata[0] == {"available": True, "price": 7.5}


def test_integer_ids_are_losslessly_normalized(inputs):
    matrix, rows, output, manifest = inputs
    large_id = 9007199254740993
    rows.write_text(json.dumps([{"id": 1}, {"id": large_id}]))
    ids = rows.with_name("ids.json")
    ids.write_text(json.dumps([large_id, "1"]))
    convert_npy(matrix, rows, output, manifest, ids_path=ids)
    assert load_dataset(output).ids == [str(large_id), "1"]


@pytest.mark.parametrize("bad_id", [1.0, True, None, "", " "])
def test_invalid_ids_rejected(inputs, bad_id):
    matrix, rows, output, manifest = inputs
    rows.write_text(json.dumps([{"id": bad_id}, {"id": "b"}]))
    with pytest.raises(DatasetValidationError, match="non-empty string or integer"):
        convert_npy(matrix, rows, output, manifest, assume_row_order=True)
    assert not output.exists()


@pytest.mark.parametrize("row_ids", [["a", "a"], [1, "1"]])
def test_duplicate_metadata_ids_rejected(inputs, row_ids):
    matrix, rows, output, manifest = inputs
    rows.write_text(json.dumps([{"id": value} for value in row_ids]))
    with pytest.raises(DatasetValidationError, match="Duplicate metadata ID"):
        convert_npy(matrix, rows, output, manifest, assume_row_order=True)


@pytest.mark.parametrize("ids_value, error", [
    (["a", "a"], "Duplicate vector ID"),
    (["a", "c"], "must match exactly"),
    (["a"], "ID count"),
])
def test_invalid_vector_ids_rejected(inputs, ids_value, error):
    matrix, rows, output, manifest = inputs
    ids = rows.with_name("ids.json")
    ids.write_text(json.dumps(ids_value))
    with pytest.raises(DatasetValidationError, match=error):
        convert_npy(matrix, rows, output, manifest, ids_path=ids)
    assert not output.exists()


def test_no_implicit_metadata_key_overwrite(inputs):
    matrix, rows, output, manifest = inputs
    rows.write_text(json.dumps([
        {"id": "a", "metadata": {"group": "x"}, "group": "y"}, {"id": "b"},
    ]))
    with pytest.raises(DatasetValidationError, match="exists in both"):
        convert_npy(matrix, rows, output, manifest, assume_row_order=True)
    assert not output.exists()


@pytest.mark.parametrize("object_source", ["matrix", "ids"])
def test_pickle_arrays_rejected(inputs, object_source):
    matrix, rows, output, manifest = inputs
    kwargs = {"assume_row_order": True}
    if object_source == "matrix":
        np.save(matrix, np.array([[1, 2], [3, 4]], dtype=object))
    else:
        ids = rows.with_name("ids.npy")
        np.save(ids, np.array(["a", "b"], dtype=object))
        kwargs = {"ids_path": ids}
    with pytest.raises(DatasetValidationError, match="Cannot read"):
        convert_npy(matrix, rows, output, manifest, **kwargs)
    assert not output.exists()


def test_json_requires_objects(inputs):
    matrix, rows, output, manifest = inputs
    rows.write_text('["a", "b"]')
    with pytest.raises(DatasetValidationError, match="row objects"):
        convert_npy(matrix, rows, output, manifest, assume_row_order=True)


def test_metadata_row_count_mismatch(inputs):
    matrix, rows, output, manifest = inputs
    rows.write_text('[{"id": "a"}]')
    with pytest.raises(DatasetValidationError, match="Metadata row count"):
        convert_npy(matrix, rows, output, manifest, assume_row_order=True)
