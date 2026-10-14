"""Slices retain row identity and distinguish JSON values from missing keys."""

from copy import deepcopy

import numpy as np
import pytest

from embedding_visualizer.compute.common import ComputeError
from embedding_visualizer.compute.filtering import apply_filters, infer_fields
from embedding_visualizer.data.models import Dataset, DatasetManifest, ValidationReport


def dataset(metadata, names=None):
    count = len(metadata)
    return Dataset(
        ids=[f"id-{index}" for index in range(count)],
        names=names if names is not None else [None] * count,
        embeddings=np.ones((count, 2)),
        metadata=metadata,
        manifest=DatasetManifest("test", "test", "1", 2),
        report=ValidationReport(count, 2),
    )


def selected(data, filters, search=""):
    mask = apply_filters(data, filters, search)
    assert mask.dtype == np.bool_
    return np.flatnonzero(mask).tolist()


def test_inference_ignores_null_for_widget_kind_and_preserves_missing_counts():
    data = dataset([
        {"n": 1, "flag": True, "group": "A", "nested": []},
        {"n": 2.5, "flag": False, "group": "B", "nested": {}},
        {"n": None, "flag": None, "group": None, "nested": None},
        {},
    ])
    fields = {item["name"]: item for item in infer_fields(data)}
    assert [item["name"] for item in infer_fields(data)] == ["flag", "group", "n", "nested"]
    assert fields["n"]["kind"] == "numeric"
    assert fields["flag"]["kind"] == "boolean"
    assert fields["group"]["kind"] == "categorical"
    assert fields["nested"]["kind"] == "mixed"
    assert fields["n"]["count"] == 3
    assert fields["n"]["missing_count"] == fields["n"]["null_count"] == 1
    assert fields["n"]["min"] == 1 and fields["n"]["max"] == 2.5
    assert fields["nested"]["type_counts"] == {"array": 1, "null": 1, "object": 1}
    assert fields["nested"]["values"] == [{"type": "null", "value": None, "count": 1}]


def test_mixed_typed_values_remain_distinct_and_choices_round_trip():
    data = dataset([{"x": True}, {"x": 1}, {"x": 1.0}, {"x": "1"}, {"x": None}, {}, {"x": []}])
    field = infer_fields(data)[0]
    assert field["kind"] == "mixed"
    assert field["value_count"] == 5
    for value, expected in [(True, [0]), (1, [1]), (1.0, [2]), ("1", [3]), (None, [4])]:
        assert selected(data, [{"field": "x", "op": "in", "values": [value]}]) == expected
    assert selected(data, [{"field": "x", "op": "in", "values": field["values"]}]) == [0, 1, 2, 3, 4]


def test_browser_integral_number_keeps_its_explicit_float_type():
    data = dataset([{"x": 1}, {"x": 1.0}, {"x": True}])
    assert selected(data, [{"field": "x", "op": "in", "values": [{"type": "number", "value": 1}]}]) == [1]


def test_no_filters_empty_selection_missing_and_null_have_explicit_semantics():
    data = dataset([{"x": None}, {}, {"x": "yes"}, {"x": {"nested": 1}}])
    assert selected(data, []) == [0, 1, 2, 3]
    assert selected(data, [{"field": "x", "op": "in", "values": []}]) == []
    assert selected(data, [{"field": "x", "op": "in", "values": [], "missing": True}]) == [1]
    assert selected(data, [{"field": "x", "op": "in", "values": [None], "missing": True}]) == [0, 1]
    assert selected(data, [{"field": "x", "op": "exists"}]) == [0, 2, 3]
    assert selected(data, [{"field": "x", "op": "missing"}]) == [1]


def test_inclusive_range_matches_numeric_values_only_with_optional_missing():
    data = dataset([{"x": False}, {"x": 0}, {"x": 1.0}, {"x": 2}, {"x": "1"}, {"x": None}, {}])
    assert selected(data, [{"field": "x", "op": "range", "min": 0, "max": 1}]) == [1, 2]
    assert selected(data, [{"field": "x", "op": "range", "min": 1, "missing": True}]) == [2, 3, 6]
    assert selected(data, [{"field": "x", "op": "range", "max": 0}]) == [1]


def test_filters_and_search_are_anded_without_mutating_data_or_filter_inputs():
    data = dataset([
        {"group": "A", "n": 2}, {"group": "A", "n": 5}, {"group": "B", "n": 6},
    ], names=["Бета", "АЛЬФА", "Альфа"])
    filters = [{"field": "group", "kind": "categorical", "op": "in", "values": ["A"]},
               {"field": "n", "kind": "numeric", "op": "range", "min": 3}]
    before_filters, before_metadata = deepcopy(filters), deepcopy(data.metadata)
    assert selected(data, filters, "альфа") == [1]
    assert selected(data, [], " ID-2 ") == [2]
    assert selected(data, filters, "Бета") == []
    assert data.metadata == before_metadata and filters == before_filters


def test_bounded_value_choices_disclose_truncation_and_are_deterministic():
    data = dataset([{"x": "z"}, {"x": "a"}, {"x": "a"}, {"x": "b"}])
    field = infer_fields(data, max_values=1)[0]
    assert field["values"] == [{"type": "string", "value": "a", "count": 2}]
    assert field["value_count"] == 3 and field["values_truncated"] is True
    assert infer_fields(data) == infer_fields(data)


@pytest.mark.parametrize("spec", [
    None, {}, {"field": "unknown", "op": "missing"}, {"field": "x", "op": "eval"},
    {"field": "x", "op": "in"}, {"field": "x", "op": "in", "values": "A"},
    {"field": "x", "op": "in", "values": [[1]]},
    {"field": "x", "op": "in", "values": [{"type": "boolean", "value": 1}]},
    {"field": "x", "op": "in", "values": [{"type": "string", "value": "a", "oops": 1}]},
    {"field": "x", "op": "in", "values": [float("nan")]},
    {"field": "x", "op": "in", "values": [], "missing": 1},
    {"field": "x", "op": "range"}, {"field": "x", "op": "range", "min": True},
    {"field": "x", "op": "range", "min": None}, {"field": "x", "op": "range", "max": float("inf")},
    {"field": "x", "op": "range", "min": 3, "max": 2},
    {"field": "x", "op": "exists", "values": []}, {"field": "x", "op": "missing", "missing": True},
    {"field": "x", "op": "exists", "kind": "date"},
])
def test_invalid_filters_fail_instead_of_silently_matching(spec):
    with pytest.raises(ComputeError):
        apply_filters(dataset([{"x": "a"}]), [spec])


def test_invalid_top_level_arguments_and_schema_have_clear_errors():
    data = dataset([{"x": "a"}])
    with pytest.raises(ComputeError, match="list"):
        apply_filters(data, {})
    with pytest.raises(ComputeError, match="string"):
        apply_filters(data, [], search=None)
    with pytest.raises(ComputeError, match="positive integer"):
        infer_fields(data, max_values=True)
    data.metadata = []
    with pytest.raises(ComputeError, match="same rows"):
        infer_fields(data)


def test_empty_dataset_and_no_metadata_fields():
    assert infer_fields(dataset([{}, {}])) == []
    assert selected(dataset([{}, {}]), []) == [0, 1]
    assert infer_fields(dataset([])) == []
    assert selected(dataset([]), []) == []


def test_large_finite_json_integers_do_not_require_lossy_float_conversion():
    large = 10 ** 400
    data = dataset([{"x": large}, {"x": large + 1}])
    assert infer_fields(data)[0]["min"] == large
    assert selected(data, [{"field": "x", "op": "range", "min": large + 1}]) == [1]
