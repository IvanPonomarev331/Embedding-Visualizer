"""Tests for the external data boundary, round trips, and safe writes."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from embedding_visualizer.data import (
    DatasetManifest,
    DatasetValidationError,
    load_dataset,
    write_dataset,
)


@pytest.fixture
def manifest():
    return DatasetManifest("Example", "model/layer", "v1", 2)


def raw_dataset(tmp_path, manifest, **columns):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest.to_dict()))
    path = tmp_path / "dataset.parquet"
    pq.write_table(pa.table(columns), path)
    return path


def test_round_trip_preserves_rows_nested_metadata_and_float32(tmp_path, manifest):
    metadata = [
        {"category": "Книги", "nested": {"values": [1, None, True]}, "weight": 1.5},
        {"category": None, "weight": 2},
    ]
    path = write_dataset(
        tmp_path / "dataset", ["a", "b"], [[1, 2], [3, 4]], manifest,
        names=["Товар", None], metadata=metadata,
    )
    loaded = load_dataset(path)
    assert loaded.ids == ["a", "b"]
    assert loaded.names == ["Товар", None]
    assert loaded.metadata == metadata
    assert loaded.manifest == manifest
    assert loaded.embeddings.dtype == np.float32
    np.testing.assert_array_equal(loaded.embeddings, [[1, 2], [3, 4]])
    assert loaded.report.row_count == 2
    assert loaded.report.metadata_fields == {
        "category": ["null", "string"], "nested": ["object"], "weight": ["integer", "number"],
    }
    schema = pq.read_schema(path / "dataset.parquet")
    assert schema.field("embedding").type == pa.list_(pa.float32(), 2)
    assert schema.field("metadata").type == pa.string()
    assert load_dataset(path / "dataset.parquet").ids == ["a", "b"]


def test_optional_columns_and_zero_vectors(tmp_path, manifest):
    path = raw_dataset(tmp_path, manifest, id=["zero", "unit"], embedding=[[0, 0], [1, 0]])
    loaded = load_dataset(path)
    assert loaded.names == [None, None]
    assert loaded.metadata == [{}, {}]
    assert loaded.report.zero_vector_ids == ["zero"]
    assert "cosine similarity is undefined" in loaded.report.warnings[0]


def test_zero_vector_with_euclidean_has_no_cosine_warning(tmp_path):
    manifest = DatasetManifest("example", "test", "v1", 2, "euclidean")
    path = write_dataset(tmp_path / "dataset", ["zero"], [[0, 0]], manifest)
    loaded = load_dataset(path)
    assert loaded.report.zero_vector_ids == ["zero"]
    assert loaded.report.warnings == []


def test_reads_struct_metadata_and_null(tmp_path, manifest):
    path = raw_dataset(
        tmp_path, manifest, id=["a", "b"], embedding=[[1, 2], [3, 4]],
        metadata=[{"kind": "new", "score": 3}, None], embedding_size=[2, 2],
    )
    assert load_dataset(path).metadata == [{"kind": "new", "score": 3}, {}]


@pytest.mark.parametrize("embeddings, expected", [
    ([[1, float("nan")]], "must be finite"),
    ([[float("inf"), 1]], "must be finite"),
    ([[1e100, 1]], "float32 range"),
    ([[1, True]], "must be a real number"),
    ([[1, "2"]], "must be a real number"),
    ([[1 + 2j, 3]], "must be a real number"),
    ([[1]], "embedding length 1"),
    ([[]], "embedding length 0"),
    (["12"], "one-dimensional numeric vector"),
    ([12], "one-dimensional numeric vector"),
    ([None], "one-dimensional numeric vector"),
    ([[[1], [2]]], "must be a real number"),
])
def test_rejects_bad_vectors_without_creating_output(tmp_path, manifest, embeddings, expected):
    output = tmp_path / "must-not-exist"
    with pytest.raises(DatasetValidationError, match=expected):
        write_dataset(output, ["bad"], embeddings, manifest)
    assert not output.exists()


@pytest.mark.parametrize("ids, embeddings, expected", [
    (["a", "a"], [[1, 2], [3, 4]], "duplicate id"),
    ([""], [[1, 2]], "nonempty string"),
    (["  "], [[1, 2]], "nonempty string"),
    ([None], [[1, 2]], "nonempty string"),
    ([1], [[1, 2]], "nonempty string"),
    ([], [], "at least one row"),
    (["a", "b"], [[1, 2]], "embedding has 1 rows; id has 2"),
])
def test_rejects_bad_ids_and_row_counts(tmp_path, manifest, ids, embeddings, expected):
    with pytest.raises(DatasetValidationError, match=expected):
        write_dataset(tmp_path / "dataset", ids, embeddings, manifest)


def test_ragged_parquet_reports_row_id_and_dimension(tmp_path, manifest):
    path = raw_dataset(tmp_path, manifest, id=["a", "bad"], embedding=[[1, 2], [3]])
    with pytest.raises(DatasetValidationError, match=r"row 2 \(id='bad'\): embedding length 1"):
        load_dataset(path)


@pytest.mark.parametrize("size", [3, None, True, 2.0])
def test_rejects_incorrect_optional_embedding_size(tmp_path, manifest, size):
    path = raw_dataset(tmp_path, manifest, id=["a"], embedding=[[1, 2]], embedding_size=[size])
    with pytest.raises(DatasetValidationError, match="embedding_size must be integer 2"):
        load_dataset(path)


@pytest.mark.parametrize("value", ["[]", "invalid json", "null", "42", {"n": float("nan")}, {"nested": {3: 4}}])
def test_rejects_nonobject_or_invalid_json_metadata(tmp_path, manifest, value):
    with pytest.raises(DatasetValidationError, match="invalid metadata"):
        write_dataset(tmp_path / "dataset", ["a"], [[1, 2]], manifest, metadata=[value])


def test_rejects_bad_optional_name_and_metadata_length(tmp_path, manifest):
    with pytest.raises(DatasetValidationError, match="name must be a string or null"):
        write_dataset(tmp_path / "dataset", ["a"], [[1, 2]], manifest, names=[5])
    with pytest.raises(DatasetValidationError, match="metadata has 0 rows; id has 1"):
        write_dataset(tmp_path / "dataset", ["a"], [[1, 2]], manifest, metadata=[])


@pytest.mark.parametrize("field, value", [
    ("dataset_name", ""), ("space_id", " "), ("version", 1),
    ("embedding_dim", 0), ("embedding_dim", True), ("embedding_dim", 2.0),
    ("default_metric", "l2"), ("schema_version", True), ("schema_version", 2),
])
def test_manifest_rejects_bad_values(manifest, field, value):
    data = manifest.to_dict()
    data[field] = value
    with pytest.raises(DatasetValidationError, match=f"manifest.{field}"):
        DatasetManifest.from_dict(data)


def test_manifest_requires_identity_and_rejects_unknown_fields(manifest):
    with pytest.raises(DatasetValidationError, match="missing fields"):
        DatasetManifest.from_dict({"dataset_name": "test"})
    with pytest.raises(DatasetValidationError, match="unknown fields"):
        DatasetManifest.from_dict({**manifest.to_dict(), "unknown": 1})
    with pytest.raises(DatasetValidationError, match="must be a JSON object"):
        DatasetManifest.from_dict([])


def test_missing_and_corrupt_files_have_useful_errors(tmp_path, manifest):
    with pytest.raises(DatasetValidationError, match="cannot read manifest"):
        load_dataset(tmp_path)
    (tmp_path / "manifest.json").write_text("{broken")
    with pytest.raises(DatasetValidationError, match="cannot read manifest"):
        load_dataset(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest.to_dict()))
    (tmp_path / "dataset.parquet").write_bytes(b"not parquet")
    with pytest.raises(DatasetValidationError, match="cannot read Parquet"):
        load_dataset(tmp_path)


def test_missing_required_column(tmp_path, manifest):
    path = raw_dataset(tmp_path, manifest, id=["a"])
    with pytest.raises(DatasetValidationError, match="missing columns: embedding"):
        load_dataset(path)


def test_warns_about_ignored_columns_with_bounded_names(tmp_path, manifest):
    extras = {f"extra_{index}": [index] for index in range(12)}
    path = raw_dataset(tmp_path, manifest, id=["a"], embedding=[[1, 2]], **extras)
    loaded = load_dataset(path)
    assert loaded.metadata == [{}]
    assert len(loaded.report.warnings) == 1
    warning = loaded.report.warnings[0]
    assert "Ignored 12 non-contract Parquet column(s)" in warning
    assert "'extra_0'" in warning
    assert "'extra_9'" in warning
    assert "'extra_10'" not in warning
    assert "2 more" in warning
    assert "Move fields to metadata" in warning


def test_diagnostic_count_is_bounded(tmp_path, manifest):
    with pytest.raises(DatasetValidationError) as caught:
        write_dataset(tmp_path / "dataset", [""] * 30, [[1, 2]] * 30, manifest)
    assert len(caught.value.issues) == 20
    assert caught.value.total_issues == 30
    assert "10 more issues" in str(caught.value)


def test_refuses_overwrite_preserves_previous_dataset_on_invalid_input(tmp_path, manifest):
    path = write_dataset(tmp_path / "dataset", ["a"], [[1, 2]], manifest)
    before = (path / "dataset.parquet").read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_dataset(path, ["b"], [[3, 4]], manifest)
    with pytest.raises(DatasetValidationError):
        write_dataset(path, ["b"], [[float("nan"), 4]], manifest, overwrite=True)
    assert (path / "dataset.parquet").read_bytes() == before
    assert load_dataset(path).ids == ["a"]


def test_explicit_overwrite_preserves_unrelated_files(tmp_path, manifest):
    path = write_dataset(tmp_path / "dataset", ["a"], [[1, 2]], manifest)
    (path / "notes.txt").write_text("keep")
    write_dataset(path, ["b"], [[3, 4]], manifest, overwrite=True)
    assert load_dataset(path).ids == ["b"]
    assert (path / "notes.txt").read_text() == "keep"


def test_rolls_back_if_second_file_commit_fails(tmp_path, manifest, monkeypatch):
    from embedding_visualizer.data import parquet

    path = write_dataset(tmp_path / "dataset", ["a"], [[1, 2]], manifest)
    original_replace = parquet.os.replace

    def fail_manifest_once(source, destination):
        if source.name == "manifest.json" and source.parent != path:
            raise OSError("simulated disk error")
        return original_replace(source, destination)

    monkeypatch.setattr(parquet.os, "replace", fail_manifest_once)
    with pytest.raises(OSError, match="simulated disk error"):
        write_dataset(path, ["b"], [[3, 4]], manifest, overwrite=True)
    assert load_dataset(path).ids == ["a"]
