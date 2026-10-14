"""One research session, independent of HTTP and browser rendering."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import threading
import uuid

import numpy as np

from .compute.common import ComputeError, prepare_embeddings
from .compute.filtering import apply_filters, infer_fields
from .compute.projections import project
from .compute.clustering import CLUSTER_METHODS, cluster
from .compute.tuning import TUNABLE_METHODS, TuningCancelled, tune_clustering
from .compute.neighbors import exact_neighbors
from .compute.summaries import summarize_clusters
from .data import load_dataset
from .persistence import ComputeCache, PersistenceError, dataset_fingerprint, export_rows, read_study_bytes, study_bytes


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name}: ожидается целое число от {low} до {high}.")
    return value


def validate_result_descriptor(value, operation, count):
    if value is None:
        return
    methods = ("pca", "umap", "tsne") if operation == "projection" else CLUSTER_METHODS
    if (not isinstance(value, dict) or not isinstance(value.get("key"), str) or not value["key"]
        or value.get("method") not in methods or value.get("scope") not in ("all", "slice")
        or not isinstance(value.get("parameters"), dict) or not isinstance(value.get("request"), dict)
        or not isinstance(value.get("warnings"), list) or any(not isinstance(x, str) for x in value["warnings"])):
        raise ValueError(f"Некорректное описание {operation} в исследовании.")
    integer(value.get("count"), f"{operation}.count", 1, count)
    if operation == "clustering" and value.get("space") not in ("original", "pca"):
        raise ValueError("Некорректное пространство кластеризации в исследовании.")
    if operation == "clustering" and value.get("tuning") is not None:
        validate_tuning_report(value["tuning"], value["count"], value["request"], value["space"])


def validate_tuning_report(report, count, request=None, space=None):
    """Bound saved search histories and reject malformed winner metadata."""
    if (not isinstance(report, dict) or report.get("objective") not in ("inertia", "silhouette")
        or report.get("score_space") not in ("original", "pca")
        or report.get("score_metric") != "euclidean" or not isinstance(report.get("trials"), list)):
        raise ValueError("Некорректный отчёт автоподбора.")
    budget = integer(report.get("budget"), "tuning.budget", 1, 48)
    total = integer(report.get("total"), "tuning.total", 1, budget)
    if len(report["trials"]) != total or report.get("completed") != total:
        raise ValueError("Неполный отчёт автоподбора.")
    integer(report.get("sample_count"), "tuning.sample_count", 1, count)
    winner = integer(report.get("best_trial"), "tuning.best_trial", 1, total)
    def finite_score(value):
        return type(value) in (float, int) and np.isfinite(value)
    for index, trial in enumerate(report["trials"], 1):
        if (not isinstance(trial, dict) or trial.get("trial") != index
            or trial.get("status") not in ("ok", "error") or not isinstance(trial.get("params"), dict)
            or (trial["status"] == "ok" and not finite_score(trial.get("score")))):
            raise ValueError("Некорректный вариант в отчёте автоподбора.")
    best = report["trials"][winner - 1]
    if (best["status"] != "ok" or not finite_score(report.get("best_score"))
        or report["best_score"] != best["score"]):
        raise ValueError("Некорректный победитель автоподбора.")
    if ((request is not None and request != best["params"])
        or (space is not None and space != report["score_space"])):
        raise ValueError("Параметры результата не совпадают с победителем автоподбора.")
    scores = [trial["score"] for trial in report["trials"] if trial["status"] == "ok"]
    if report["best_score"] != (min(scores) if report["objective"] == "inertia" else max(scores)):
        raise ValueError("Победитель не имеет лучшей оценки автоподбора.")


class Workspace:
    def __init__(self, cache_dir):
        self.cache = ComputeCache(Path(cache_dir) / "results")
        self.lock = threading.RLock()
        self.dataset = None
        self.source = None
        self.fingerprint = None
        self.matrix = None
        self.normalized = False
        self.filters = []
        self.selected_ids = []
        self.primary = None
        self.view = {}
        self.projection = None
        self.clustering = None
        self.summaries = []
        self.job = None
        self.cancel_event = None
        self.generation = 0
        self.selection_revision = -1

    def require_dataset(self):
        if self.dataset is None:
            raise ValueError("Сначала откройте набор данных.")
        return self.dataset

    def idle(self):
        if self.job and self.job["status"] == "running":
            raise ValueError("Дождитесь текущего вычисления.")

    def load(self, path):
        # Validate before replacing a working session.
        resolved = Path(path).expanduser().resolve()
        dataset = load_dataset(resolved)
        fingerprint = dataset_fingerprint(dataset)
        matrix = prepare_embeddings(dataset.embeddings)
        fields = infer_fields(dataset)
        with self.lock:
            self.idle()
            self.dataset, self.source, self.fingerprint = dataset, str(resolved), fingerprint
            self.matrix, self.fields = matrix, fields
            self.id_index = {identifier: index for index, identifier in enumerate(dataset.ids)}
            self.normalized, self.filters, self.view = False, [], {}
            self.mask = np.ones(len(dataset.ids), dtype=bool)
            self.primary, self.selected_ids = dataset.ids[0], [dataset.ids[0]]
            self.projection, self.clustering, self.summaries, self.job = None, None, [], None
            self.generation += 1
            return self.info()

    def info(self):
        with self.lock:
            dataset = self.require_dataset()
            projection = None if self.projection is None else {k: v for k, v in self.projection.items() if k != "coordinates"}
            clustering = None if self.clustering is None else {k: v for k, v in self.clustering.items() if k != "labels"}
            return json_safe(dict(source=self.source, fingerprint=self.fingerprint,
                manifest=dataset.manifest.to_dict(), count=len(dataset.ids), visible_count=int(self.mask.sum()),
                fields=self.fields, validation=asdict(dataset.report), normalized=self.normalized,
                filters=self.filters, primary=self.primary, selected_ids=self.selected_ids,
                projection=projection, clustering=clustering, summaries=self.summaries, view=self.view,
                dimensions=self.coordinates.shape[1], geometry=self.geometry))

    @property
    def coordinates(self):
        return self.matrix if self.projection is None else self.projection["coordinates"]

    @property
    def geometry(self):
        return f"{self.fingerprint}:{self.normalized}:" + ("raw" if self.projection is None else self.projection["key"])

    def set_normalization(self, value):
        if type(value) is not bool:
            raise ValueError("normalize должен быть boolean.")
        with self.lock:
            self.idle()
            dataset = self.require_dataset()
            if value != self.normalized:
                self.matrix = prepare_embeddings(dataset.embeddings, normalize=value)
                self.normalized = value
                self.projection, self.clustering, self.summaries = None, None, []
                self.generation += 1
            return self.info()

    def set_filters(self, filters):
        with self.lock:
            dataset = self.require_dataset()
            mask = apply_filters(dataset, filters)
            self.mask, self.filters = mask, filters
            return self.info()

    def select(self, ids, primary=None, revision=None, fingerprint=None):
        with self.lock:
            self.require_dataset()
            if fingerprint is not None and fingerprint != self.fingerprint:
                raise ValueError("Набор данных изменился; повторите выбор объекта.")
            if revision is not None:
                integer(revision, "selection.revision", 0, 2**53 - 1)
                if revision < self.selection_revision:
                    return {"primary": self.primary, "selected_ids": self.selected_ids}
            if not isinstance(ids, list) or any(not isinstance(x, str) or x not in self.id_index for x in ids):
                raise ValueError("Выделение содержит неизвестный ID.")
            if primary is not None and primary not in self.id_index:
                raise ValueError("Неизвестный ID объекта.")
            self.selected_ids = list(dict.fromkeys(ids))
            self.primary = primary if primary is not None else (self.selected_ids[-1] if self.selected_ids else self.primary)
            if revision is not None:
                self.selection_revision = revision
            return {"primary": self.primary, "selected_ids": self.selected_ids}

    def objects(self, query="", offset=0, limit=100):
        with self.lock:
            dataset = self.require_dataset()
            integer(offset, "offset", 0, len(dataset.ids))
            integer(limit, "limit", 1, 500)
            rows = np.flatnonzero(self.mask & apply_filters(dataset, [], search=query))
            return {"total": len(rows), "offset": offset, "rows": [self.row(int(i)) for i in rows[offset:offset + limit]]}

    def row(self, index):
        dataset = self.dataset
        label = None if self.clustering is None else int(self.clustering["labels"][index])
        return dict(id=dataset.ids[index], name=dataset.names[index], metadata=dataset.metadata[index],
                    cluster=None if label == -2 else label)

    def map_data(self, x=0, y=1, color="", include_id=None):
        with self.lock:
            dataset = self.require_dataset()
            coordinates = self.coordinates
            integer(x, "Ось X", 0, coordinates.shape[1] - 1)
            integer(y, "Ось Y", 0, coordinates.shape[1] - 1)
            candidates = np.flatnonzero(self.mask & np.isfinite(coordinates[:, x]) & np.isfinite(coordinates[:, y]))
            indices = candidates
            if len(indices) > 50_000:
                indices = np.sort(np.random.default_rng(42).choice(indices, 50_000, replace=False))
            if include_id in self.id_index:
                index = self.id_index[include_id]
                if index in candidates and index not in indices:
                    indices = np.append(indices, index)
            values, kind = [], "none"
            if color == "__cluster__" and self.clustering is not None:
                values = [int(self.clustering["labels"][i]) for i in indices]
                kind = "cluster"
            elif color:
                field = next((f for f in self.fields if f["name"] == color), None)
                if field is None:
                    raise ValueError("Неизвестное поле окраски.")
                values = [dataset.metadata[i].get(color) for i in indices]
                kind = "numeric" if field["kind"] == "numeric" else "categorical"
            return json_safe(dict(ids=[dataset.ids[i] for i in indices], names=[dataset.names[i] for i in indices],
                x=coordinates[indices, x], y=coordinates[indices, y], values=values, color_kind=kind,
                clusters=None if self.clustering is None else self.clustering["labels"][indices],
                missing=[color not in dataset.metadata[i] for i in indices] if color and color != "__cluster__" else [],
                shown=len(indices), available=len(candidates), filtered=int(self.mask.sum()), total=len(dataset.ids),
                geometry=self.geometry, projection="raw" if self.projection is None else self.projection["method"]))

    def object_detail(self, identifier, k=10, metric="cosine", scope="all"):
        with self.lock:
            dataset = self.require_dataset()
            if identifier not in self.id_index:
                raise ValueError("Объект не найден.")
            integer(k, "k", 1, 1000)
            if scope not in ("all", "slice"):
                raise ValueError("Область поиска должна быть all или slice.")
            index = self.id_index[identifier]
            indices = np.arange(len(dataset.ids)) if scope == "all" else np.flatnonzero(self.mask)
            candidate_count = int(len(indices) - int(index in indices))
            if index not in indices:
                indices = np.append(indices, index)
            position = int(np.flatnonzero(indices == index)[0])
            matrix = self.matrix if scope == "all" else self.matrix[indices]
            # The operation uses a stable snapshot while other view requests stay responsive.
            row = self.row(index)
            row.update(norm=float(np.linalg.norm(dataset.embeddings[index].astype(float))),
                       original_vector=dataset.embeddings[index].tolist(), vector=self.matrix[index].tolist(),
                       coordinates=json_safe(self.coordinates[index]),
                       in_slice=bool(self.mask[index]), candidate_count=candidate_count)
        try:
            result = exact_neighbors(matrix, position, k=k, metric=metric)
            row["neighbors"] = [dict(id=dataset.ids[int(indices[i])], name=dataset.names[int(indices[i])], value=float(value))
                                for i, value in zip(result.indices, result.values, strict=True)]
            row.update(value_kind=result.value_kind, excluded_zero_count=result.excluded_zero_count)
        except ComputeError as exc:
            row.update(neighbors=[], warning=str(exc))
        return row

    def start_operation(self, operation, method, params=None, scope="all", tuning=None):
        with self.lock:
            self.idle()
            self.require_dataset()
            if operation not in ("projection", "clustering") or scope not in ("all", "slice"):
                raise ValueError("Неизвестная операция или область вычислений.")
            if not isinstance(params, dict):
                raise ValueError("params должен быть объектом.")
            if tuning is not None:
                if operation != "clustering" or method not in TUNABLE_METHODS:
                    raise ValueError("Автоподбор с заданным числом кластеров доступен для KMeans, Spectral, Agglomerative и Gaussian Mixture. DBSCAN/HDBSCAN определяют число кластеров сами.")
                if not isinstance(tuning, dict) or set(tuning) - {"objective", "budget"}:
                    raise ValueError("tuning должен содержать только objective и budget.")
                tuning = {"objective": tuning.get("objective", "inertia"),
                          "budget": integer(tuning.get("budget", 12), "budget", 1, 48)}
                if tuning["objective"] not in ("inertia", "silhouette"):
                    raise ValueError("Критерий автоподбора: inertia или silhouette.")
                if set(params) - {"n_clusters", "space", "pca_components", "random_state"}:
                    raise ValueError("В автоподборе параметры алгоритма выбираются автоматически; задайте только число кластеров, пространство и seed.")
            if operation == "projection" and method == "raw":
                self.projection = None
                self.job = dict(id=uuid.uuid4().hex, status="done", cache_hit=False)
                return self.job.copy()
            indices = np.arange(len(self.matrix)) if scope == "all" else np.flatnonzero(self.mask)
            if not len(indices):
                raise ValueError("Текущий срез пуст. Измените фильтры.")
            cache_params = {"method": method, "normalized": self.normalized, "params": params}
            if tuning is not None:
                cache_params["tuning"] = tuning
            key = self.cache.key(self.fingerprint, "clustering-search:1" if tuning is not None else operation + ":2",
                cache_params, indices)
            self.job = dict(id=uuid.uuid4().hex, status="running", operation=operation, method=method)
            self.cancel_event = threading.Event() if tuning is not None else None
            if tuning is not None:
                self.job.update(cancellable=True, cancel_requested=False,
                                progress={"completed": 0, "total": tuning["budget"], "best_score": None})
            snapshot = (self.job["id"], self.generation, self.dataset, self.matrix, indices, key)
            threading.Thread(target=self._compute, args=(snapshot, operation, method, dict(params), scope, tuning, self.cancel_event), daemon=True).start()
            return self.job.copy()

    def cancel_job(self):
        with self.lock:
            if not self.job or self.job["status"] != "running":
                return (self.job or {"status": "idle"}).copy()
            if self.cancel_event is None or self.job.get("operation") != "clustering" or not (
                self.job.get("cancellable") or self.job.get("cancel_requested")
            ):
                raise ValueError("Отмена доступна для автоподбора кластеризации.")
            self.cancel_event.set()
            self.job.update(cancel_requested=True, cancellable=False)
            return self.job.copy()

    def _compute(self, snapshot, operation, method, params, scope, tuning=None, cancel_event=None):
        job_id, generation, dataset, matrix, indices, key = snapshot
        def check_cancelled():
            if cancel_event is not None and cancel_event.is_set():
                raise TuningCancelled("Автоподбор отменён. Предыдущая кластеризация сохранена.")
        def progress(value):
            with self.lock:
                if self.job and self.job["id"] == job_id and self.job["status"] == "running":
                    self.job["progress"] = json_safe(value)
        try:
            check_cancelled()
            cached = self.cache.load(key)
            hit = cached is not None
            if cached is None:
                vectors = matrix[indices]
                if operation == "projection":
                    result = project(vectors, method, **params)
                    arrays = {"coordinates": result.coordinates}
                    metadata = dict(parameters=result.parameters, warnings=result.warnings,
                                    variance=json_safe(result.explained_variance_ratio))
                else:
                    search = None
                    if tuning is not None:
                        search = tune_clustering(vectors, method, **params, **tuning,
                            progress=progress, cancelled=cancel_event.is_set)
                        result = search.result
                        check_cancelled()
                    else:
                        result = cluster(vectors, method, **params)
                    from .data.models import Dataset
                    subset = Dataset([dataset.ids[i] for i in indices], [dataset.names[i] for i in indices],
                        dataset.embeddings[indices], [dataset.metadata[i] for i in indices], dataset.manifest, dataset.report)
                    summaries = summarize_clusters(subset, result.labels, result.representation)
                    arrays = {"labels": result.labels}
                    metadata = dict(parameters=result.parameters, warnings=result.warnings, space=result.space,
                                    summaries=summaries)
                    if search is not None:
                        metadata.update(tuning=search.report, tuning_request=tuning,
                                        winning_request=search.request)
                        validate_tuning_report(search.report, len(indices))
                check_cancelled()
                try:
                    self.cache.save(key, arrays, metadata)
                except (OSError, PersistenceError) as exc:
                    metadata["warnings"] = metadata["warnings"] + [f"Результат готов, но не сохранён в кеш: {exc}"]
            else:
                arrays, metadata = cached["arrays"], cached["metadata"]
            record = dict({k: v for k, v in metadata.items() if k not in ("summaries", "winning_request")}, key=key, method=method,
                          scope=scope, count=len(indices), request=metadata.get("winning_request", params), cache_hit=hit)
            if tuning is not None:
                validate_tuning_report(record.get("tuning"), len(indices), record["request"], record["space"])
            if operation == "projection":
                part = arrays["coordinates"]
                if part.ndim != 2 or len(part) != len(indices) or not np.isfinite(part).all():
                    raise ValueError("Повреждён результат проекции в кеше.")
                coordinates = np.full((len(matrix), part.shape[1]), np.nan)
                coordinates[indices] = part
                record["coordinates"] = coordinates
            else:
                part = arrays["labels"]
                if part.shape != (len(indices),) or part.dtype.kind not in "iu" or np.any(part < -1):
                    raise ValueError("Повреждены метки кластеров в кеше.")
                if tuning is not None and (np.any(part < 0) or len(np.unique(part)) != record["request"]["n_clusters"]):
                    raise ValueError("Метки не соответствуют заданному числу кластеров автоподбора.")
                labels = np.full(len(matrix), -2, dtype=np.int64)
                labels[indices] = part
                record["labels"] = labels
                summaries = metadata["summaries"]
            with self.lock:
                if self.generation != generation or self.job["id"] != job_id:
                    return
                check_cancelled()
                if operation == "projection":
                    self.projection = record
                else:
                    self.clustering, self.summaries = record, summaries
                self.job.update(status="done", cache_hit=hit, cancellable=False)
        except TuningCancelled:
            with self.lock:
                if self.job and self.job["id"] == job_id:
                    self.job.update(status="cancelled", cancellable=False)
        except Exception as exc:
            with self.lock:
                if self.job and self.job["id"] == job_id:
                    self.job.update(status="cancelled" if cancel_event is not None and cancel_event.is_set() else "error",
                                    error=str(exc), cancellable=False)

    def clear_clusters(self):
        with self.lock:
            self.idle()
            self.clustering, self.summaries = None, []
            return self.info()

    def save(self, view):
        with self.lock:
            self.idle()
            state = self.info()
            if not isinstance(view, dict):
                raise ValueError("view должен быть объектом.")
            state["view"] = view
            state["study_schema"] = 1
            arrays = {}
            if self.projection is not None:
                arrays["coordinates"] = self.projection["coordinates"]
            if self.clustering is not None:
                arrays["labels"] = self.clustering["labels"]
            return study_bytes(state, arrays)

    def restore(self, data):
        stored = read_study_bytes(data)
        state, arrays = stored["state"], stored["arrays"]
        if state.get("study_schema") != 1:
            raise ValueError("Неподдерживаемая версия исследования.")
        with self.lock:
            self.idle()
            if self.dataset is not None and self.fingerprint == state.get("fingerprint"):
                dataset, source = self.dataset, self.source
            else:
                source = state["source"]
                dataset = load_dataset(Path(source).expanduser())
                if dataset_fingerprint(dataset) != state.get("fingerprint"):
                    raise ValueError("Данные изменились: снимок не совпадает с сохранённым исследованием. Откройте исходный снимок.")
            normalized = state["normalized"]
            matrix = prepare_embeddings(dataset.embeddings, normalize=normalized)
            mask = apply_filters(dataset, state["filters"])
            id_index = {identifier: i for i, identifier in enumerate(dataset.ids)}
            selected = state["selected_ids"]
            if not isinstance(selected, list) or any(x not in id_index for x in selected) or state["primary"] not in id_index:
                raise ValueError("Исследование содержит неизвестные ID.")
            projection, clustering = state["projection"], state["clustering"]
            validate_result_descriptor(projection, "projection", len(dataset.ids))
            validate_result_descriptor(clustering, "clustering", len(dataset.ids))
            if projection is not None:
                coordinates = arrays.get("coordinates")
                if (coordinates is None or coordinates.ndim != 2 or coordinates.shape[0] != len(dataset.ids)
                    or coordinates.shape[1] < 1 or coordinates.dtype.kind not in "fiu" or np.isinf(coordinates).any()
                    or np.any(np.isfinite(coordinates).any(axis=1) != np.isfinite(coordinates).all(axis=1))):
                    raise ValueError("Некорректные координаты в исследовании.")
                if int(np.isfinite(coordinates).all(axis=1).sum()) != projection["count"]:
                    raise ValueError("Число рассчитанных координат не совпадает с описанием проекции.")
                projection = dict(projection, coordinates=coordinates)
            if clustering is not None:
                labels = arrays.get("labels")
                if labels is None or labels.shape != (len(dataset.ids),) or labels.dtype.kind not in "iu" or np.any(labels < -2):
                    raise ValueError("Некорректные метки кластеров в исследовании.")
                if int((labels != -2).sum()) != clustering["count"]:
                    raise ValueError("Число кластеризованных объектов не совпадает с описанием.")
                if clustering.get("tuning") is not None and (
                    np.any(labels == -1) or len(np.unique(labels[labels >= 0])) != clustering["request"]["n_clusters"]
                ):
                    raise ValueError("Метки не соответствуют заданному числу кластеров автоподбора.")
                clustering = dict(clustering, labels=labels)
            summaries = state.get("summaries", [])
            if not isinstance(summaries, list) or not isinstance(state.get("view", {}), dict):
                raise ValueError("Некорректные настройки отображения или сводки кластеров.")
            for summary in summaries:
                if (not isinstance(summary, dict) or clustering is None
                    or type(summary.get("cluster_id")) is not int or not isinstance(summary.get("metadata"), dict)
                    or not isinstance(summary.get("example_ids"), list)
                    or any(identifier not in id_index for identifier in summary["example_ids"])
                    or summary.get("size") != int((clustering["labels"] == summary["cluster_id"]).sum())):
                    raise ValueError("Повреждена сводка кластеров в исследовании.")
            self.dataset, self.source, self.fingerprint = dataset, source, state["fingerprint"]
            self.matrix, self.normalized, self.mask, self.filters = matrix, normalized, mask, state["filters"]
            self.id_index, self.fields = id_index, infer_fields(dataset)
            self.primary, self.selected_ids = state["primary"], selected
            self.projection, self.clustering, self.summaries = projection, clustering, summaries
            self.view, self.job = state.get("view", {}), None
            self.generation += 1
            return self.info()

    def export(self, scope="selected", format="csv"):
        with self.lock:
            dataset = self.require_dataset()
            if scope == "selected":
                indices = np.array([self.id_index[x] for x in self.selected_ids], dtype=int)
            elif scope == "slice":
                indices = np.flatnonzero(self.mask)
            elif scope == "all":
                indices = np.arange(len(dataset.ids))
            else:
                raise ValueError("Область экспорта: selected, slice или all.")
            labels = None if self.clustering is None else self.clustering["labels"]
            return export_rows(dataset, indices, coordinates=self.coordinates, labels=labels, format=format)
