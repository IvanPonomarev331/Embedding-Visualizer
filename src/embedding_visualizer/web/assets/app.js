/* Local embedding workspace. Selection updates existing traces; only changes
 * to geometry, axes, filters or color request and redraw the map. */
"use strict";

const $ = (id) => document.getElementById(id);
const fmt = new Intl.NumberFormat("ru-RU");
const METHODS = {raw: "Координаты", pca: "PCA", umap: "UMAP", tsne: "t-SNE", kmeans: "KMeans", hdbscan: "HDBSCAN", spectral: "Spectral", agglomerative: "Иерархическая", dbscan: "DBSCAN", gaussian_mixture: "Gaussian mixture"};
const TUNABLE_METHODS = ["kmeans", "spectral", "agglomerative", "gaussian_mixture"];
const TUNING_OBJECTIVES = {inertia: "Разброс ↓", silhouette: "Silhouette ↑"};
const TYPES = {string: "строка", boolean: "логическое", integer: "целое", number: "число", null: "null"};
const BASE_COLOR = "#61b5ff";
const NOISE_COLOR = "#929dab";
const PALETTE = ["#62b2ff", "#ffbd59", "#47e4a1", "#ff6b8a", "#be95ff", "#30d8ee", "#ff8d4b", "#eff36d", "#f082dc", "#a1e46b", "#7c91ff", "#f1d2a2", "#00bea4", "#ff6770", "#d2b6ff", "#79daef", "#d9a256", "#bddb54", "#d768b3", "#8ee2be"];
const PLOT_THEMES = {
  dark: {background: "#111722", grid: "#253044", tick: "#899ab2", text: "#c6d3e5", panel: "#1b2433", border: "#3c4a60", outline: "#f4f8ff"},
  light: {background: "#f6f8fc", grid: "#e1e7f0", tick: "#77879b", text: "#435873", panel: "#ffffff", border: "#cdd7e5", outline: "#182c43"},
};
function storedPreference(key, fallback) { try { return localStorage.getItem(key) || fallback; } catch (_) { return fallback; } }
function savePreference(key, value) { try { localStorage.setItem(key, value); } catch (_) { /* Preferences are optional. */ } }
const initialTheme = storedPreference("visualizer.theme", "dark") === "light" ? "light" : "dark";
const initialPointSize = Math.max(3, Math.min(16, Number(storedPreference("visualizer.pointSize", "8")) || 8));
const CONTROL_IDS = ["projection-method", "pca-components", "umap-neighbors", "umap-min-dist", "tsne-perplexity", "tsne-iterations", "projection-metric", "projection-seed", "projection-scope", "cluster-method", "cluster-space", "cluster-pca-components", "cluster-count", "cluster-seed", "hdbscan-size", "hdbscan-samples", "cluster-metric", "cluster-scope", "spectral-neighbors", "cluster-linkage", "dbscan-eps", "dbscan-samples", "gmm-covariance", "gmm-iterations", "gmm-regularization", "cluster-mode", "cluster-objective", "cluster-budget"];
const S = {
  token: "", workspace: "", config: null, info: null, map: null, detail: null,
  filters: [], busy: false, replacingDataset: false, job: null, jobKind: null, jobTuning: null, cancelRequested: false, plotReady: false, plotKey: "", restoreSerial: 0,
  view: {x: 0, y: 1, color: "", viewport: null, mode: "pan", pointSize: initialPointSize, theme: initialTheme},
  mapSequence: 0, detailSequence: 0, searchSequence: 0, objectSequence: 0, selectionSequence: 0,
  selectionQueue: Promise.resolve(), mutationQueue: Promise.resolve(), drawer: null, objectOffset: 0, objectTotal: 0,
  searchRows: [], searchActive: -1, pollTimer: null, rendering: false,
};

function node(tag, className, text) {
  const result = document.createElement(tag);
  if (className) result.className = className;
  if (text !== undefined) result.textContent = text;
  return result;
}

function option(value, label) {
  const result = node("option", "", label);
  result.value = value;
  return result;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[char]);
}

function compactNumber(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  return Math.abs(value) >= 1e5 || Math.abs(value) < 1e-3 ? value.toExponential(3) : Number(value.toPrecision(5)).toString();
}

function toast(message, error = false) {
  const item = node("div", `toast${error ? " error" : ""}`);
  item.append(node("span", "", String(message)));
  const close = node("button", "", "×");
  close.setAttribute("aria-label", "Закрыть сообщение");
  close.onclick = () => item.remove();
  item.append(close);
  $("toasts").append(item);
  while ($("toasts").children.length > 4) $("toasts").firstElementChild.remove();
  setTimeout(() => item.remove(), error ? 14000 : 5000);
}

function on(id, event, callback) {
  $(id).addEventListener(event, (e) => {
    try { Promise.resolve(callback(e)).catch((error) => toast(error.message || error, true)); }
    catch (error) { toast(error.message || error, true); }
  });
}

async function request(path, body, options = {}) {
  const headers = {"X-Visualizer-Token": S.token, "X-Visualizer-Workspace": S.workspace};
  let payload;
  if (body !== undefined) {
    headers["Content-Type"] = options.raw ? "application/octet-stream" : "application/json";
    payload = options.raw ? body : JSON.stringify(body);
  }
  const response = await fetch(path, {method: body === undefined ? "GET" : "POST", headers, body: payload, cache: "no-store"});
  if (!response.ok) {
    let message = `Ошибка ${response.status}`;
    try { const error = await response.json(); message = error.error || error.message || message; } catch (_) { /* HTTP status remains useful. */ }
    throw new Error(message);
  }
  return options.binary ? response : response.json();
}

function mutation(path, body, options) {
  const pending = S.mutationQueue.then(() => request(path, body, options));
  S.mutationQueue = pending.then(() => undefined, () => undefined);
  return pending;
}

async function download(path, body, fallback) {
  const response = await request(path, body, {binary: true});
  const blob = await response.blob();
  const disposition = response.headers.get("Content-Disposition") || "";
  const extended = disposition.match(/filename\*=UTF-8''([^;]+)/i);
  const simple = disposition.match(/filename="?([^";]+)"?/i);
  let filename = extended ? decodeURIComponent(extended[1]) : simple ? simple[1] : fallback;
  filename = filename.replace(/[\\/]/g, "_");
  const url = URL.createObjectURL(blob);
  const link = node("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function fieldByName(name) {
  return S.info?.fields.find((field) => field.name === name);
}

function typedKey(value, missing = false) {
  return missing ? "missing" : `${value === null ? "null" : typeof value}:${JSON.stringify(value)}`;
}

function choiceLabel(choice) {
  const value = choice.value === null ? "null" : typeof choice.value === "string" ? choice.value || '""' : String(choice.value);
  return `${value} · ${TYPES[choice.type] || choice.type}${choice.count !== undefined ? ` (${fmt.format(choice.count)})` : ""}`;
}

function sameChoice(left, right) {
  return left.type === right.type && JSON.stringify(left.value) === JSON.stringify(right.value);
}

function normalizeChoice(value) {
  if (value && typeof value === "object" && "type" in value) return value;
  const type = value === null ? "null" : typeof value === "boolean" ? "boolean" : typeof value === "string" ? "string" : Number.isInteger(value) ? "integer" : "number";
  return {type, value};
}

function updateParameterVisibility() {
  const method = $("cluster-method").value;
  const tunable = TUNABLE_METHODS.includes(method);
  if (!tunable) $("cluster-mode").value = "manual";
  const auto = $("cluster-mode").value === "auto";
  $("cluster-mode").querySelector('option[value="auto"]').disabled = !tunable;
  $("cluster-mode-note").hidden = tunable;
  document.querySelectorAll("[data-projection]").forEach((element) => { element.hidden = !element.dataset.projection.split(" ").includes($("projection-method").value); });
  document.querySelectorAll("[data-cluster]").forEach((element) => { element.hidden = !element.dataset.cluster.split(" ").includes(method) || (auto && element.hasAttribute("data-cluster-manual")); });
  document.querySelectorAll("[data-cluster-auto]").forEach((element) => { element.hidden = !auto; });
  $("cluster-seed-label").hidden = !auto && !["kmeans", "spectral", "gaussian_mixture"].includes(method);
  $("run-cluster").textContent = auto ? "Подобрать параметры" : "Найти кластеры";
  $("cluster-pca-label").hidden = $("cluster-space").value !== "pca";
  const ward = $("cluster-method").value === "agglomerative" && $("cluster-linkage").value === "ward";
  $("cluster-metric").querySelector('option[value="cosine"]').disabled = ward;
  if (ward) $("cluster-metric").value = "euclidean";
  updateBounds();
  syncDisabled();
}

function setBound(id, maximum, initial, reset = false) {
  const input = $(id);
  input.max = Math.max(Number(input.min) || 0, maximum);
  if (reset || Number(input.value) > Number(input.max) || Number(input.value) < Number(input.min)) input.value = Math.min(initial, Number(input.max));
}

function updateBounds(reset = false) {
  if (!S.info) return;
  const dim = S.info.manifest.embedding_dim;
  const projectionCount = $("projection-scope").value === "slice" ? S.info.visible_count : S.info.count;
  const clusterCount = $("cluster-scope").value === "slice" ? S.info.visible_count : S.info.count;
  setBound("pca-components", Math.max(1, Math.min(dim, projectionCount)), 2, reset);
  setBound("umap-neighbors", Math.max(2, projectionCount - 1), 15, reset);
  setBound("tsne-perplexity", Math.max(.1, projectionCount - .01), Math.min(30, Math.max(.1, (projectionCount - 1) / 3)), reset);
  setBound("cluster-pca-components", Math.max(1, Math.min(dim, clusterCount)), 10, reset);
  const method = $("cluster-method").value;
  const maximumClusters = method === "spectral" ? Math.min(256, clusterCount - 1) : method === "gaussian_mixture" ? Math.min(256, clusterCount) : clusterCount;
  setBound("cluster-count", Math.max(1, maximumClusters), 3, reset);
  setBound("spectral-neighbors", Math.max(2, Math.min(1000, clusterCount)), 15, reset);
  setBound("dbscan-samples", Math.max(1, clusterCount), 5, reset);
  setBound("hdbscan-size", Math.max(2, clusterCount), 5, reset);
  setBound("hdbscan-samples", Math.max(0, clusterCount), 0, reset);
}

function syncDisabled() {
  const loaded = Boolean(S.info);
  for (const id of ["run-projection", "run-cluster", "normalize", "save-study"]) $(id).disabled = !loaded || S.busy;
  $("load-dataset").disabled = S.busy;
  $("open-study").disabled = S.busy;
  $("clear-clusters").disabled = !S.info?.clustering || S.busy;
  for (const id of ["search", "axis-x", "axis-y", "color-field", "export-toggle", "apply-filters", "reset-filters", "focus-selection", "reset-view"]) $(id).disabled = !loaded;
  $("add-filter").disabled = !S.info?.fields.length;
  for (const formId of ["projection-form", "cluster-form"]) {
    $(formId).querySelectorAll("input,select,button").forEach((element) => {
      element.disabled = !loaded || S.busy || Boolean(element.closest("[hidden]"));
    });
  }
  $("clear-clusters").disabled = !S.info?.clustering || S.busy;
  if ($("yt-form")) $("yt-form").querySelectorAll("input,select,button").forEach((element) => { element.disabled = S.busy || !S.config?.yt_available; });
}

function setBusy(value, message = "Вычисление…") {
  S.busy = value;
  $("job-status").hidden = !value;
  $("job-message").textContent = message;
  if (!value) $("cancel-job").hidden = true;
  syncDisabled();
}

function applyClusterControls(clustering) {
  $("cluster-method").value = clustering?.method || "kmeans";
  $("cluster-scope").value = clustering?.scope || "all";
  if (!clustering) return;
  const p = clustering.request || {};
  const mapping = {space: "cluster-space", pca_components: "cluster-pca-components", n_clusters: "cluster-count", random_state: "cluster-seed", min_cluster_size: "hdbscan-size", min_samples: clustering.method === "dbscan" ? "dbscan-samples" : "hdbscan-samples", metric: "cluster-metric", n_neighbors: "spectral-neighbors", linkage: "cluster-linkage", eps: "dbscan-eps", covariance_type: "gmm-covariance", max_iter: "gmm-iterations", reg_covar: "gmm-regularization"};
  Object.entries(mapping).forEach(([key, id]) => { if (p[key] !== undefined) $(id).value = p[key] ?? 0; });
}

function restoreControls(info) {
  const projection = info.projection;
  const clustering = info.clustering;
  $("projection-method").value = projection?.method || "raw";
  $("projection-scope").value = projection?.scope || "all";
  if (projection) {
    const p = projection.request || {};
    const mapping = {n_components: "pca-components", n_neighbors: "umap-neighbors", min_dist: "umap-min-dist", perplexity: "tsne-perplexity", max_iter: "tsne-iterations", metric: "projection-metric", random_state: "projection-seed"};
    Object.entries(mapping).forEach(([key, id]) => { if (p[key] !== undefined) $(id).value = p[key]; });
  }
  applyClusterControls(clustering);
  $("cluster-mode").value = clustering?.tuning ? "auto" : "manual";
  if (clustering?.tuning) {
    const tuning = clustering.tuning_request || clustering.tuning;
    $("cluster-objective").value = tuning.objective || "inertia";
    $("cluster-budget").value = tuning.budget || 12;
    if (clustering.tuning.base_seed !== undefined) $("cluster-seed").value = clustering.tuning.base_seed;
  }
  const saved = info.view?.controls;
  if (saved && typeof saved === "object") CONTROL_IDS.forEach((id) => { if (saved[id] !== undefined) $(id).value = saved[id]; });
}

function updateInfo(info, {reset = false, restore = false, filters = false} = {}) {
  const isNew = !S.info || S.info.fingerprint !== info.fingerprint;
  S.info = info;
  if (reset || isNew) {
    S.mapSequence += 1;
    S.detailSequence += 1;
    S.searchSequence += 1;
    S.objectSequence += 1;
    S.selectionSequence += 1;
    S.view = {x: 0, y: Math.min(1, info.dimensions - 1), color: "", viewport: null, mode: "pan", pointSize: S.view.pointSize, theme: S.view.theme};
    S.detail = null;
    S.objectOffset = 0;
    $("neighbor-metric").value = info.manifest.default_metric;
    $("neighbor-scope").value = "all";
    $("neighbor-k").value = Math.min(10, Math.max(1, info.count - 1));
    $("search").value = "";
    $("search-results").hidden = true;
    $("projection-method").value = "raw";
    $("projection-scope").value = "all";
    $("cluster-method").value = "kmeans";
    $("cluster-mode").value = "manual";
    $("cluster-objective").value = "inertia";
    $("cluster-budget").value = "12";
    $("cluster-space").value = "original";
    $("cluster-scope").value = "all";
    updateBounds(true);
  }
  if (restore) {
    const view = info.view || {};
    S.view.x = Number.isInteger(view.axes?.[0]) ? view.axes[0] : 0;
    S.view.y = Number.isInteger(view.axes?.[1]) ? view.axes[1] : Math.min(1, info.dimensions - 1);
    S.view.color = typeof view.color === "string" ? view.color : "";
    S.view.viewport = validViewport(view.viewport) ? view.viewport : null;
    S.view.mode = ["pan", "select", "lasso"].includes(view.mode) ? view.mode : "pan";
    if (Number.isFinite(view.pointSize)) S.view.pointSize = Math.max(3, Math.min(16, view.pointSize));
    if (["dark", "light"].includes(view.theme)) S.view.theme = view.theme;
    applyThemeStyles();
    updatePointSizeControl();
    if (["cosine", "euclidean", "dot_product"].includes(view.neighborMetric)) $("neighbor-metric").value = view.neighborMetric;
    if (["all", "slice"].includes(view.neighborScope)) $("neighbor-scope").value = view.neighborScope;
    if (Number.isInteger(view.k) && view.k >= 1 && view.k <= 1000) $("neighbor-k").value = view.k;
    restoreControls(info);
    S.restoreSerial += 1;
  }
  if (reset || restore || isNew || filters) {
    S.filters = structuredClone(info.filters || []);
    renderFilters();
  }
  S.view.x = Math.max(0, Math.min(S.view.x, info.dimensions - 1));
  S.view.y = Math.max(0, Math.min(S.view.y, info.dimensions - 1));
  $("dataset-name").textContent = info.manifest.dataset_name;
  $("dataset-name").title = info.source;
  $("dataset-meta").textContent = `${fmt.format(info.count)} объектов · ${info.manifest.embedding_dim} измерений`;
  $("dataset-path").value = info.source;
  const datasetList = $("dataset-list");
  if (![...datasetList.options].some((entry) => entry.value === info.source)) {
    datasetList.append(option(info.source, info.manifest.dataset_name));
  }
  datasetList.value = info.source;
  $("dataset-list-label").hidden = false;
  $("dataset-dot").classList.add("ready");
  $("normalize").checked = info.normalized;
  $("empty-state").hidden = true;
  $("projection-badge").textContent = METHODS[info.projection?.method || "raw"];
  $("cluster-badge").textContent = info.clustering ? METHODS[info.clustering.method] : "Нет";
  $("projection-detail").textContent = info.projection ? `${fmt.format(info.projection.count)} объектов · ${info.projection.scope === "slice" ? "срез при запуске" : "весь набор"}${info.projection.cache_hit ? " · из кеша" : ""}` : (info.normalized ? "Координаты нормализованных векторов." : "Координаты исходных векторов.");
  $("cluster-detail").textContent = info.clustering ? `${fmt.format(info.clustering.count)} объектов · ${info.clustering.space === "pca" ? "PCA" : "до проекции карты"}${info.clustering.cache_hit ? " · из кеша" : ""}` : "";
  renderTuning();
  for (const [prefix, result] of [["projection", info.projection], ["cluster", info.clustering]]) {
    $(`${prefix}-result`).hidden = !result;
    $(`${prefix}-parameters`).textContent = result ? JSON.stringify(result.parameters, null, 2) : "";
    $(`${prefix}-warnings`).textContent = (result?.warnings || []).join(" ");
  }
  $("projection-variance").textContent = info.projection?.variance?.length
    ? `Объяснённая дисперсия: ${(100 * info.projection.variance.reduce((sum, value) => sum + value, 0)).toFixed(1)}%. По компонентам: ${info.projection.variance.map((value) => `${(100 * value).toFixed(1)}%`).join(", ")}.`
    : "";
  $("filter-badge").textContent = info.filters.length ? `${fmt.format(info.visible_count)} / ${fmt.format(info.count)}` : "Все объекты";
  $("objects-count").textContent = fmt.format(info.visible_count);
  $("clusters-count").textContent = info.summaries?.length || "0";
  const warnings = info.validation?.warnings || [];
  $("dataset-warning").hidden = !warnings.length;
  $("dataset-warning").textContent = warnings.join(" ");
  for (const [id, value] of [["axis-x", S.view.x], ["axis-y", S.view.y]]) {
    $(id).replaceChildren(...Array.from({length: info.dimensions}, (_, index) => option(index, String(index))));
    $(id).value = value;
  }
  const colors = [option("", "Один цвет")];
  if (info.clustering) colors.push(option("__cluster__", "Кластеры"));
  info.fields.forEach((field) => colors.push(option(field.name, field.name)));
  $("color-field").replaceChildren(...colors);
  if (!colors.some((item) => item.value === S.view.color)) S.view.color = "";
  $("color-field").value = S.view.color;
  updateBounds();
  updateParameterVisibility();
  updateSelectionBadge();
  updateModeButtons();
  if (S.drawer === "clusters") renderClusters();
  syncDisabled();
}

async function refreshAll({restore = false} = {}) {
  await Promise.all([refreshMap({restore}), refreshObject(), refreshObjectsIfOpen()]);
}

async function loadDataset(path) {
  if (!path.trim()) throw new Error("Укажите путь к набору данных.");
  setBusy(true, "Чтение набора…");
  S.replacingDataset = true;
  try {
    await S.selectionQueue.catch(() => {});
    S.selectionQueue = Promise.resolve();
    const info = await mutation("/api/load", {path: path.trim()});
    updateInfo(info, {reset: true, filters: true});
    await refreshAll();
    closeMobilePanels();
  } finally { S.replacingDataset = false; setBusy(false); }
}

function numericInput(id) {
  const value = Number($(id).value);
  if (!$(id).value.trim() || !Number.isFinite(value)) throw new Error("Заполните числовые параметры вычисления.");
  return value;
}

function projectionRequest() {
  const method = $("projection-method").value;
  const params = {};
  if (method === "pca") params.n_components = numericInput("pca-components");
  if (method === "umap") Object.assign(params, {n_components: 2, n_neighbors: numericInput("umap-neighbors"), min_dist: numericInput("umap-min-dist"), metric: $("projection-metric").value});
  if (method === "tsne") Object.assign(params, {n_components: 2, perplexity: numericInput("tsne-perplexity"), max_iter: numericInput("tsne-iterations"), metric: $("projection-metric").value});
  if (method !== "raw") params.random_state = numericInput("projection-seed");
  return {operation: "projection", method, params, scope: $("projection-scope").value};
}

function clusterRequest() {
  const method = $("cluster-method").value;
  const params = {space: $("cluster-space").value};
  if (params.space === "pca") params.pca_components = numericInput("cluster-pca-components");
  if ($("cluster-mode").value === "auto" && TUNABLE_METHODS.includes(method)) {
    Object.assign(params, {n_clusters: numericInput("cluster-count"), random_state: numericInput("cluster-seed")});
    return {operation: "clustering", method, params, scope: $("cluster-scope").value,
      tuning: {objective: $("cluster-objective").value, budget: numericInput("cluster-budget")}};
  }
  if (["kmeans", "spectral", "agglomerative", "gaussian_mixture"].includes(method)) params.n_clusters = numericInput("cluster-count");
  if (["kmeans", "spectral", "gaussian_mixture"].includes(method)) params.random_state = numericInput("cluster-seed");
  if (method === "spectral") params.n_neighbors = numericInput("spectral-neighbors");
  if (method === "agglomerative") Object.assign(params, {linkage: $("cluster-linkage").value, metric: $("cluster-metric").value});
  if (method === "hdbscan") Object.assign(params, {min_cluster_size: numericInput("hdbscan-size"), min_samples: numericInput("hdbscan-samples") || null, metric: $("cluster-metric").value});
  if (method === "dbscan") Object.assign(params, {eps: numericInput("dbscan-eps"), min_samples: numericInput("dbscan-samples"), metric: $("cluster-metric").value});
  if (method === "gaussian_mixture") Object.assign(params, {covariance_type: $("gmm-covariance").value, max_iter: numericInput("gmm-iterations"), reg_covar: numericInput("gmm-regularization")});
  return {operation: "clustering", method, params, scope: $("cluster-scope").value};
}

async function startOperation(payload) {
  S.cancelRequested = false;
  S.jobTuning = payload.tuning || null;
  setBusy(true, `${METHODS[payload.method] || payload.method} · вычисление…`);
  S.jobKind = payload.operation;
  try {
    S.job = await mutation("/api/operation", payload);
    await pollJob();
  } catch (error) { S.job = null; S.jobKind = null; S.jobTuning = null; setBusy(false); throw error; }
}

function showJobProgress(job) {
  const progress = job.progress;
  const cancelled = S.cancelRequested || job.cancel_requested;
  const parts = [S.jobTuning || progress ? "Автоподбор" : "Вычисление"];
  if (progress) {
    parts.push(`${progress.completed ?? 0} / ${progress.total ?? "…"}`);
    if (Number.isFinite(progress.best_score)) parts.push(`лучшее: ${compactNumber(progress.best_score)}`);
  }
  if (cancelled) parts.push("Отмена после текущего варианта…");
  else if (progress?.trial) {
    const trialNumber = typeof progress.trial === "object" ? progress.trial.trial : progress.trial;
    if (Number.isFinite(trialNumber)) parts.push(`вариант ${trialNumber}`);
  }
  $("job-message").textContent = parts.join(" · ");
  $("cancel-job").hidden = !job.cancellable && !cancelled;
  $("cancel-job").disabled = cancelled || !job.cancellable;
  $("cancel-job").textContent = cancelled ? "Отменяется…" : "Отменить";
}

async function cancelJob() {
  if (!S.job?.cancellable || S.cancelRequested) return;
  S.cancelRequested = true;
  showJobProgress(S.job);
  try {
    const job = await mutation("/api/job/cancel", {});
    if (S.job && job.id === S.job.id) { S.job = job; showJobProgress(job); }
  } catch (error) { S.cancelRequested = false; if (S.job) showJobProgress(S.job); throw error; }
}

async function pollJob() {
  clearTimeout(S.pollTimer);
  try {
    const job = await request("/api/job");
    if (S.job && job.id !== S.job.id) return;
    if (job.status === "running") { S.job = job; showJobProgress(job); S.pollTimer = setTimeout(pollJob, 400); return; }
    const kind = S.jobKind;
    S.job = null;
    S.jobKind = null;
    S.jobTuning = null;
    S.cancelRequested = false;
    setBusy(false);
    if (job.status === "error") { toast(job.error || "Вычисление завершилось с ошибкой.", true); return; }
    if (job.status === "cancelled") { toast("Подбор отменён. Предыдущие кластеры сохранены."); return; }
    const info = await request("/api/state");
    if (kind === "clustering") S.view.color = "__cluster__";
    updateInfo(info, {reset: kind === "import", filters: kind === "import"});
    await refreshAll();
    const result = kind === "projection" ? info.projection : kind === "clustering" ? info.clustering : null;
    (result?.warnings || []).forEach((warning) => toast(warning));
    if (kind === "import") toast("Набор импортирован и открыт.");
  } catch (error) { setBusy(false); toast(error.message, true); }
}

function makeLabel(caption, input) {
  const label = node("label", "", caption);
  label.append(input);
  return label;
}

function renderFilters() {
  const container = $("filter-list");
  container.replaceChildren();
  if (!S.filters.length) {
    container.append(node("p", "filter-empty", S.info?.fields.length ? "Фильтры по полям metadata" : "В наборе нет полей metadata."));
    return;
  }
  S.filters.forEach((spec, index) => {
    const field = fieldByName(spec.field);
    if (!field) return;
    const item = node("div", "filter-item");
    const header = node("div", "filter-header");
    const fieldSelect = node("select");
    fieldSelect.setAttribute("aria-label", `Поле фильтра ${index + 1}`);
    S.info.fields.forEach((entry) => fieldSelect.append(option(entry.name, entry.name)));
    fieldSelect.value = spec.field;
    fieldSelect.onchange = () => {
      const newField = fieldByName(fieldSelect.value);
      S.filters[index] = newFilter(newField);
      renderFilters();
    };
    const remove = node("button", "icon-button", "×");
    remove.title = "Удалить фильтр";
    remove.setAttribute("aria-label", "Удалить фильтр");
    remove.onclick = () => { S.filters.splice(index, 1); renderFilters(); };
    header.append(fieldSelect, remove);
    item.append(header);
    const operation = node("select");
    operation.setAttribute("aria-label", "Условие фильтра");
    operation.append(option("in", "Одно из значений"));
    if (field.min !== null) operation.append(option("range", "В диапазоне"));
    operation.append(option("exists", "Поле присутствует"), option("missing", "Поле отсутствует"));
    operation.value = spec.op;
    operation.onchange = () => {
      S.filters[index] = {field: field.name, kind: field.kind, op: operation.value};
      if (operation.value === "in") S.filters[index].values = [];
      if (operation.value === "range") Object.assign(S.filters[index], {min: field.min, max: field.max});
      renderFilters();
    };
    item.append(operation);
    if (spec.op === "range") {
      const bounds = node("div", "two-columns");
      for (const [bound, caption] of [["min", "От"], ["max", "До"]]) {
        const input = node("input"); input.type = "number"; input.step = "any";
        input.placeholder = compactNumber(field[bound]);
        input.value = spec[bound] ?? "";
        input.oninput = () => { if (!input.value.trim()) delete spec[bound]; else spec[bound] = Number(input.value); };
        bounds.append(makeLabel(caption, input));
      }
      item.append(bounds);
    } else if (spec.op === "in") {
      const values = (spec.values || []).map(normalizeChoice);
      const choices = [...field.values];
      values.forEach((value) => { if (!choices.some((entry) => sameChoice(entry, value))) choices.push(value); });
      const select = node("select"); select.multiple = true;
      select.setAttribute("aria-label", `Значения поля ${field.name}`);
      choices.forEach((choice, i) => { const entry = option(i, choiceLabel(choice)); entry.selected = values.some((value) => sameChoice(choice, value)); select.append(entry); });
      select.onchange = () => { spec.values = Array.from(select.selectedOptions, (entry) => ({type: choices[entry.value].type, value: choices[entry.value].value})); };
      item.append(select);
      item.append(node("p", "hint", "Пустой выбор не включает ни одного значения. null — отдельное значение."));
      if (field.values_truncated) item.append(node("p", "hint", `Частые значения: ${field.values.length} из ${fmt.format(field.value_count)}.`));
      const custom = node("details", "custom-filter-value");
      custom.append(node("summary", "", "Указать значение вручную"));
      const customBody = node("div", "custom-filter-body");
      const typeSelect = node("select");
      Object.entries(TYPES).forEach(([type, label]) => typeSelect.append(option(type, label)));
      typeSelect.value = field.kind === "boolean" ? "boolean" : field.kind === "numeric" ? "number" : "string";
      const input = node("input"); input.type = "text"; input.placeholder = "Точное значение";
      input.setAttribute("aria-label", `Своё значение поля ${field.name}`);
      typeSelect.setAttribute("aria-label", "Тип своего значения");
      typeSelect.onchange = () => { input.disabled = typeSelect.value === "null"; input.placeholder = typeSelect.value === "boolean" ? "true или false" : "Точное значение"; };
      const add = node("button", "outline-button", "Добавить к выбору");
      const addValue = () => {
        try {
          const choice = parseCustomChoice(typeSelect.value, input.value);
          if (!(spec.values || []).map(normalizeChoice).some((entry) => sameChoice(entry, choice))) spec.values = [...(spec.values || []), choice];
          renderFilters();
        } catch (error) { toast(error.message, true); }
      };
      add.onclick = addValue;
      input.onkeydown = (event) => { if (event.key === "Enter") { event.preventDefault(); addValue(); } };
      customBody.append(typeSelect, input, add); custom.append(customBody); item.append(custom);
    }
    if (spec.op === "in" || spec.op === "range") {
      const label = node("label", "check-label");
      const check = node("input"); check.type = "checkbox"; check.checked = Boolean(spec.missing);
      check.onchange = () => { spec.missing = check.checked; };
      label.append(check, node("span", "", `Включить отсутствие поля (${fmt.format(field.missing_count)})`));
      item.append(label);
    }
    container.append(item);
  });
}

function newFilter(field) {
  if (field.kind === "numeric") return {field: field.name, kind: field.kind, op: "range", min: field.min, max: field.max};
  return {field: field.name, kind: field.kind, op: "in", values: []};
}

function parseCustomChoice(type, text) {
  if (type === "string") return {type, value: text};
  if (type === "null") return {type, value: null};
  if (type === "boolean") {
    if (!["true", "false"].includes(text.trim())) throw new Error("Для логического значения укажите true или false.");
    return {type, value: text.trim() === "true"};
  }
  const value = Number(text);
  if (!text.trim() || !Number.isFinite(value)) throw new Error("Укажите конечное число.");
  if (type === "integer" && !Number.isSafeInteger(value)) throw new Error("Для целого значения укажите целое число в точном диапазоне браузера (±9 007 199 254 740 991).");
  if (!["integer", "number"].includes(type)) throw new Error("Неизвестный тип значения.");
  return {type, value};
}

async function applyFilters(filters) {
  const info = await mutation("/api/filters", {filters: structuredClone(filters)});
  S.objectOffset = 0;
  updateInfo(info, {filters: true});
  await refreshAll();
}

function hash(value) {
  let result = 2166136261;
  for (const char of value) result = Math.imul(result ^ char.charCodeAt(0), 16777619);
  return result >>> 0;
}

function numericColor(value, low, high) {
  const colors = [[101,94,239], [42,164,221], [38,212,174], [247,226,102]];
  const t = high === low ? .5 : Math.max(0, Math.min(1, (value - low) / (high - low)));
  const part = Math.min(2, Math.floor(t * 3));
  const ratio = t * 3 - part;
  return `rgb(${colors[part].map((v, i) => Math.round(v + (colors[part + 1][i] - v) * ratio)).join(",")})`;
}

function mapColors(data) {
  const legend = $("legend");
  legend.replaceChildren();
  legend.hidden = data.color_kind === "none";
  if (data.color_kind === "none") return BASE_COLOR;
  if (data.color_kind === "numeric") {
    const finite = data.values.filter((value) => typeof value === "number" && Number.isFinite(value));
    let low = Infinity, high = -Infinity;
    finite.forEach((value) => { low = Math.min(low, value); high = Math.max(high, value); });
    if (finite.length) legend.append(node("span", "", S.view.color), node("span", "", compactNumber(low)), node("span", "legend-gradient"), node("span", "", compactNumber(high)));
    if (finite.length < data.values.length) addLegend("Нет числового значения", NOISE_COLOR);
    return data.values.map((value) => typeof value === "number" && Number.isFinite(value) ? numericColor(value, low, high) : NOISE_COLOR);
  }
  const categories = new Map();
  const colors = data.values.map((value, index) => {
    const missing = Boolean(data.missing?.[index]);
    const key = typedKey(value, missing);
    let label, color;
    if (data.color_kind === "cluster") {
      label = value === -1 ? "Шум" : value === -2 ? "Без кластера" : `Кластер ${value}`;
      color = clusterColor(value);
    } else {
      label = missing ? "Нет поля" : value === null ? "null" : typeof value === "object" ? JSON.stringify(value) : String(value);
      color = missing || value === null ? NOISE_COLOR : PALETTE[hash(key) % PALETTE.length];
    }
    if (!categories.has(key)) categories.set(key, {label, color, count: 0});
    categories.get(key).count++;
    return color;
  });
  const values = [...categories.values()].sort((a, b) => b.count - a.count);
  const legendLimit = data.color_kind === "cluster" ? 256 : 20;
  values.slice(0, legendLimit).forEach((entry) => addLegend(`${entry.label} · ${fmt.format(entry.count)}`, entry.color));
  if (values.length > legendLimit) legend.append(node("span", "", `Ещё ${values.length - legendLimit}`));
  return colors;
}

function addLegend(label, color) {
  const entry = node("span", "legend-item");
  const dot = node("i", "legend-dot"); dot.style.background = color;
  const caption = node("span", "", label); caption.title = label;
  entry.append(dot, caption); $("legend").append(entry);
}

function extent(values) {
  let low = Infinity, high = -Infinity;
  values.forEach((value) => { if (Number.isFinite(value)) { low = Math.min(low, value); high = Math.max(high, value); } });
  if (!Number.isFinite(low)) return [-1, 1];
  const padding = high === low ? Math.max(1, Math.abs(low) * .05) : (high - low) * .045;
  return [low - padding, high + padding];
}

function validViewport(viewport) {
  return viewport && ["x", "y"].every((axis) => Array.isArray(viewport[axis]) && viewport[axis].length === 2 && viewport[axis].every(Number.isFinite) && viewport[axis][0] < viewport[axis][1]);
}

async function refreshMap({restore = false} = {}) {
  if (!S.info) return;
  const sequence = ++S.mapSequence;
  const data = await request("/api/map", {x: S.view.x, y: S.view.y, color: S.view.color, include_id: S.info.primary});
  if (sequence !== S.mapSequence) return;
  const key = `${data.geometry}|${S.view.x}|${S.view.y}|${S.restoreSerial}`;
  const ranges = ((restore || key === S.plotKey) && validViewport(S.view.viewport)) ? S.view.viewport : {x: extent(data.x), y: extent(data.y)};
  S.map = data;
  S.mapIndex = new Map(data.ids.map((identifier, index) => [identifier, index]));
  S.view.viewport = {x: [...ranges.x], y: [...ranges.y]};
  S.baseColors = mapColors(data);
  const style = selectionStyle(data);
  const customdata = data.ids.map((identifier, index) => [identifier, escapeHtml(data.names[index] || ""), escapeHtml(identifier)]);
  const trace = {type: "scattergl", mode: "markers", x: data.x, y: data.y, customdata,
    marker: {color: style.colors, size: style.sizes, opacity: .96, line: {width: style.widths, color: PLOT_THEMES[S.view.theme].outline}},
    selectedpoints: style.positions.length ? style.positions : null,
    selected: {marker: {opacity: 1}}, unselected: {marker: {opacity: .9}},
    hovertemplate: "%{customdata[1]}<br>ID: %{customdata[2]}<extra></extra>"};
  const theme = PLOT_THEMES[S.view.theme];
  const axisName = data.projection === "raw" ? "Координата" : "Компонента";
  const axis = {zeroline: false, gridcolor: theme.grid, gridwidth: 1, showline: false, tickfont: {size: 11, color: theme.tick}, nticks: 8, automargin: false, autorange: false, fixedrange: false};
  const layout = {autosize: true, margin: {l: 52, r: 22, t: 26, b: 67}, paper_bgcolor: theme.background, plot_bgcolor: theme.background, showlegend: false,
    uirevision: key, selectionrevision: S.selectionSequence, dragmode: S.view.mode, clickmode: "event", hovermode: "closest",
    xaxis: {...axis, range: [...ranges.x], title: {text: `${axisName} ${S.view.x}`, font: {size: 12, color: theme.tick}, standoff: 9}},
    yaxis: {...axis, range: [...ranges.y], title: {text: `${axisName} ${S.view.y}`, font: {size: 12, color: theme.tick}, standoff: 8}},
    font: {family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'},
    hoverlabel: {bgcolor: theme.panel, bordercolor: theme.border, font: {size: 13, color: theme.text}}};
  S.rendering = true;
  try {
    await Plotly.react($("plot"), [trace], layout, {responsive: true, displayModeBar: false, scrollZoom: true, doubleClick: false, displaylogo: false});
    if (!S.plotReady) bindPlotEvents();
    S.plotReady = true;
    S.plotKey = key;
    await highlightSelection();
  } finally { S.rendering = false; }
  $("map-no-points").hidden = data.shown !== 0;
  let status = `${METHODS[data.projection]} · ${fmt.format(data.shown)} на карте`;
  if (data.available > data.shown) status += ` / ${fmt.format(data.available)} · выборка`;
  if (data.filtered !== data.total) status += ` · срез ${fmt.format(data.filtered)} / ${fmt.format(data.total)}`;
  if (data.filtered > data.available) status += ` · без координат: ${fmt.format(data.filtered - data.available)}`;
  $("map-status").textContent = status;
  $("map-status").title = "Проекции, кластеры и соседи вычисляются на выбранной области. Карта показывает до 50 000 точек.";
}

function bindPlotEvents() {
  const plot = $("plot");
  plot.on("plotly_click", (event) => {
    const identifier = event.points?.[0]?.customdata?.[0];
    if (!identifier) return;
    const extend = event.event?.shiftKey || event.event?.metaKey || event.event?.ctrlKey;
    let ids = extend ? [...S.info.selected_ids] : [];
    if (extend && ids.includes(identifier)) ids = ids.filter((id) => id !== identifier); else ids.push(identifier);
    setSelection(ids, identifier).catch((error) => toast(error.message, true));
  });
  plot.on("plotly_selected", (event) => {
    if (S.rendering || !event) return;
    const ids = [...new Set((event.points || []).map((point) => point.customdata?.[0]).filter(Boolean))];
    setSelection(ids, ids.at(-1)).catch((error) => toast(error.message, true));
  });
  plot.on("plotly_deselect", () => { if (!S.rendering) setSelection([]).catch((error) => toast(error.message, true)); });
  plot.on("plotly_relayout", () => {
    if (S.rendering || !plot._fullLayout) return;
    const candidate = {x: [...plot._fullLayout.xaxis.range], y: [...plot._fullLayout.yaxis.range]};
    if (validViewport(candidate)) S.view.viewport = candidate;
  });
}

function clusterColor(value) {
  if (value === -1) return NOISE_COLOR;
  if (!Number.isInteger(value) || value < 0) return BASE_COLOR;
  if (value < PALETTE.length) return PALETTE[value];
  return `hsl(${Math.round(value * 137.508) % 360}, 78%, ${value % 2 ? 66 : 76}%)`;
}

function selectionStyle(data) {
  const selected = new Set(S.info.selected_ids);
  const positions = [], sizes = [], colors = [], widths = [];
  data.ids.forEach((identifier, index) => {
    const active = selected.has(identifier), primary = active && identifier === S.info.primary;
    if (active) positions.push(index);
    sizes.push(S.view.pointSize + (primary ? 6 : active ? 3 : 0));
    // Selection preserves cluster identity even while the map is colored by metadata.
    colors.push(active ? clusterColor(data.clusters?.[index]) : Array.isArray(S.baseColors) ? S.baseColors[index] : S.baseColors || BASE_COLOR);
    widths.push(primary ? 2.5 : active ? 1.5 : 0);
  });
  return {positions, sizes, colors, widths};
}

async function highlightSelection() {
  if (!S.plotReady || !S.map || !S.info) return;
  const style = selectionStyle(S.map);
  await Plotly.restyle($("plot"), {selectedpoints: [style.positions.length ? style.positions : null], "marker.size": [style.sizes], "marker.color": [style.colors], "marker.line.width": [style.widths], "marker.line.color": PLOT_THEMES[S.view.theme].outline}, [0]);
}

function updatePointSizeControl() {
  $("point-size").value = S.view.pointSize;
  $("point-size-value").textContent = S.view.pointSize;
}

function applyThemeStyles() {
  document.documentElement.dataset.theme = S.view.theme;
  const caption = S.view.theme === "dark" ? "Светлая тема" : "Тёмная тема";
  $("toggle-theme").textContent = S.view.theme === "dark" ? "☀" : "☾";
  $("toggle-theme").title = caption;
  $("toggle-theme").setAttribute("aria-label", caption);
}

async function toggleTheme() {
  S.view.theme = S.view.theme === "dark" ? "light" : "dark";
  savePreference("visualizer.theme", S.view.theme);
  applyThemeStyles();
  if (!S.plotReady) return;
  const theme = PLOT_THEMES[S.view.theme];
  await Plotly.relayout($("plot"), {paper_bgcolor: theme.background, plot_bgcolor: theme.background,
    "xaxis.gridcolor": theme.grid, "yaxis.gridcolor": theme.grid,
    "xaxis.tickfont.color": theme.tick, "yaxis.tickfont.color": theme.tick,
    "xaxis.title.font.color": theme.tick, "yaxis.title.font.color": theme.tick,
    "hoverlabel.bgcolor": theme.panel, "hoverlabel.bordercolor": theme.border, "hoverlabel.font.color": theme.text});
  await highlightSelection();
}

function updateSelectionBadge() {
  const count = S.info?.selected_ids.length || 0;
  $("selection-count").hidden = !count;
  $("selection-count").textContent = `${fmt.format(count)} выделено ×`;
  document.querySelectorAll("#objects-body tr[data-id]").forEach((row) => {
    const selected = S.info.selected_ids.includes(row.dataset.id);
    row.classList.toggle("is-selected", selected);
    const check = row.querySelector("input[type=checkbox]"); if (check) check.checked = selected;
  });
}

async function setSelection(ids, primary) {
  if (!S.info || S.replacingDataset) return;
  ids = [...new Set(ids)];
  const sequence = ++S.selectionSequence;
  const fingerprint = S.info.fingerprint;
  S.info.selected_ids = ids;
  if (primary) S.info.primary = primary;
  updateSelectionBadge();
  const payload = {ids, revision: sequence, fingerprint};
  if (primary) payload.primary = primary;
  S.selectionQueue = S.selectionQueue.catch(() => {}).then(() => request("/api/selection", payload));
  try {
    const [result] = await Promise.all([S.selectionQueue, refreshObject(), highlightSelection()]);
    if (sequence === S.selectionSequence && fingerprint === S.info.fingerprint) {
      S.info.selected_ids = result.selected_ids;
      S.info.primary = result.primary;
      if (primary && S.detail?.id === primary && S.detail.in_slice && !S.mapIndex?.has(primary)
          && Number.isFinite(S.detail.coordinates?.[S.view.x]) && Number.isFinite(S.detail.coordinates?.[S.view.y])) {
        await refreshMap();
      }
    }
  } catch (error) {
    if (sequence === S.selectionSequence && fingerprint === S.info.fingerprint) throw error;
  }
}

function updateModeButtons() {
  document.querySelectorAll("[data-mode]").forEach((button) => button.classList.toggle("active", button.dataset.mode === S.view.mode));
}

async function setMode(mode) {
  S.view.mode = mode;
  updateModeButtons();
  if (S.plotReady) await Plotly.relayout($("plot"), {dragmode: mode});
}

async function focusPrimary() {
  if (!S.info || !S.plotReady) return;
  if (S.detail?.id !== S.info.primary) await refreshObject();
  const coordinates = S.detail?.coordinates;
  const index = S.mapIndex?.get(S.info.primary);
  const x = coordinates?.[S.view.x] ?? (index === undefined ? null : S.map.x[index]);
  const y = coordinates?.[S.view.y] ?? (index === undefined ? null : S.map.y[index]);
  if (!Number.isFinite(x) || !Number.isFinite(y)) { toast("У объекта нет координат в текущей проекции."); return; }
  if (!S.detail.in_slice) { toast("Объект скрыт фильтром. Измените срез, чтобы увидеть его."); return; }
  const viewport = S.view.viewport || {x: extent(S.map.x), y: extent(S.map.y)};
  const dx = (viewport.x[1] - viewport.x[0]) / 2, dy = (viewport.y[1] - viewport.y[0]) / 2;
  await Plotly.relayout($("plot"), {"xaxis.range": [x - dx, x + dx], "yaxis.range": [y - dy, y + dy], "xaxis.autorange": false, "yaxis.autorange": false});
}

async function resetViewport() {
  if (!S.plotReady || !S.map) return;
  await Plotly.relayout($("plot"), {"xaxis.range": extent(S.map.x), "yaxis.range": extent(S.map.y), "xaxis.autorange": false, "yaxis.autorange": false});
}

async function refreshObject() {
  if (!S.info?.primary) return;
  const sequence = ++S.detailSequence;
  const params = new URLSearchParams({id: S.info.primary, k: $("neighbor-k").value || "10", metric: $("neighbor-metric").value, scope: $("neighbor-scope").value});
  const detail = await request(`/api/object?${params}`);
  if (sequence !== S.detailSequence || detail.id !== S.info.primary) return;
  S.detail = detail;
  renderObject(detail);
  await highlightSelection();
}

function renderObject(detail) {
  $("inspector-empty").hidden = true;
  $("object-content").hidden = false;
  $("object-id").textContent = detail.id;
  $("object-name").textContent = detail.name || "Без названия";
  $("object-metrics").textContent = `${detail.original_vector?.length || S.info.manifest.embedding_dim} измерений · исходная норма ${compactNumber(detail.norm)}`;
  $("object-metadata").textContent = JSON.stringify(detail.metadata, null, 2);
  $("object-cluster").hidden = detail.cluster === null || detail.cluster === undefined;
  $("object-cluster").textContent = detail.cluster === -1 ? "Шум" : `Кластер ${detail.cluster}`;
  $("object-cluster").style.setProperty("--cluster-color", clusterColor(detail.cluster));
  $("object-outside").hidden = detail.in_slice;
  $("vector-body").replaceChildren();
  (detail.original_vector || []).forEach((value, index) => {
    const row = node("tr");
    row.append(node("td", "", index), node("td", "", compactNumber(value)), node("td", "", compactNumber(detail.vector?.[index])));
    $("vector-body").append(row);
  });
  const noun = detail.value_kind === "distance" ? "Расстояние: меньше — ближе." : "Близость: больше — ближе.";
  $("neighbor-caption").textContent = `${fmt.format(detail.candidate_count || 0)} кандидатов. ${noun}${detail.excluded_zero_count ? ` Нулевых векторов исключено: ${detail.excluded_zero_count}.` : ""}`;
  $("neighbor-warning").hidden = !detail.warning;
  $("neighbor-warning").textContent = detail.warning || "";
  $("neighbors-list").replaceChildren();
  (detail.neighbors || []).forEach((neighbor, index) => {
    const button = node("button", "neighbor-row");
    const title = node("span", "neighbor-title");
    title.append(node("strong", "", neighbor.name || neighbor.id), node("code", "", neighbor.id));
    button.append(node("span", "neighbor-rank", index + 1), title, node("span", "neighbor-value", compactNumber(neighbor.value)));
    button.title = `${neighbor.name || neighbor.id}\n${neighbor.id}`;
    button.onclick = () => setSelection([neighbor.id], neighbor.id).catch((error) => toast(error.message, true));
    $("neighbors-list").append(button);
  });
  if (!detail.neighbors?.length) $("neighbors-list").append(node("p", "panel-empty", "Подходящих соседей нет."));
}

async function searchObjects() {
  if (!S.info) return;
  const sequence = ++S.searchSequence;
  const query = $("search").value;
  const data = await request(`/api/objects?${new URLSearchParams({query, offset: "0", limit: "20"})}`);
  if (sequence !== S.searchSequence || document.activeElement !== $("search")) return;
  S.searchRows = data.rows;
  S.searchActive = -1;
  const menu = $("search-results"); menu.replaceChildren(); menu.hidden = false;
  if (!data.rows.length) { menu.append(node("div", "search-result-hint", "В текущем срезе ничего не найдено.")); return; }
  data.rows.forEach((row, index) => {
    const button = node("button", "search-result"); button.setAttribute("role", "option");
    button.append(node("strong", "", row.name || row.id), node("code", "", row.id));
    button.onmousedown = (event) => event.preventDefault();
    button.onclick = () => chooseSearch(index);
    menu.append(button);
  });
  if (data.total > data.rows.length) menu.append(node("div", "search-result-hint", `${fmt.format(data.total)} результатов · уточните запрос`));
}

function chooseSearch(index) {
  const row = S.searchRows[index];
  if (!row) return;
  $("search-results").hidden = true;
  $("search").blur();
  setSelection([row.id], row.id).catch((error) => toast(error.message, true));
}

function searchKeyboard(event) {
  if (!["ArrowDown", "ArrowUp", "Enter", "Escape"].includes(event.key)) return;
  event.preventDefault();
  if (event.key === "Escape") { $("search-results").hidden = true; $("search").blur(); return; }
  if (event.key === "Enter") { chooseSearch(Math.max(0, S.searchActive)); return; }
  if (!S.searchRows.length) return;
  const delta = event.key === "ArrowDown" ? 1 : -1;
  S.searchActive = (S.searchActive + delta + S.searchRows.length) % S.searchRows.length;
  $("search-results").querySelectorAll(".search-result").forEach((element, index) => {
    element.classList.toggle("active", index === S.searchActive);
    if (index === S.searchActive) element.scrollIntoView({block: "nearest"});
  });
}

async function toggleDrawer(tab) {
  S.drawer = S.drawer === tab ? null : tab;
  $("drawer-content").hidden = !S.drawer;
  $("drawer-close").hidden = !S.drawer;
  $("objects-panel").hidden = S.drawer !== "objects";
  $("clusters-panel").hidden = S.drawer !== "clusters";
  $("tuning-panel").hidden = S.drawer !== "tuning";
  document.querySelectorAll("[data-drawer]").forEach((button) => button.classList.toggle("active", button.dataset.drawer === S.drawer));
  if (S.drawer === "objects") await refreshObjects();
  if (S.drawer === "clusters") renderClusters();
  if (S.drawer === "tuning") renderTuning();
  resizePlot();
}

async function refreshObjectsIfOpen() {
  if (S.drawer === "objects") await refreshObjects();
}

async function refreshObjects() {
  if (!S.info) return;
  const sequence = ++S.objectSequence;
  const data = await request(`/api/objects?${new URLSearchParams({offset: String(S.objectOffset), limit: "100"})}`);
  if (sequence !== S.objectSequence) return;
  S.objectTotal = data.total;
  const body = $("objects-body"); body.replaceChildren();
  data.rows.forEach((row) => {
    const tr = node("tr"); tr.dataset.id = row.id; tr.tabIndex = 0;
    const checkCell = node("td", "check-cell");
    const check = node("input"); check.type = "checkbox"; check.setAttribute("aria-label", `Выделить ${row.id}`);
    check.onclick = (event) => event.stopPropagation();
    check.onchange = () => {
      const ids = check.checked ? [...S.info.selected_ids, row.id] : S.info.selected_ids.filter((id) => id !== row.id);
      setSelection(ids, row.id).catch((error) => toast(error.message, true));
    };
    checkCell.append(check);
    const metadata = node("td", "", JSON.stringify(row.metadata)); metadata.title = JSON.stringify(row.metadata, null, 2);
    tr.append(checkCell, node("td", "", row.id), node("td", "", row.name || "—"), node("td", "", row.cluster === null ? "—" : row.cluster === -1 ? "Шум" : row.cluster), metadata);
    const pick = () => setSelection([row.id], row.id).catch((error) => toast(error.message, true));
    tr.onclick = pick;
    tr.onkeydown = (event) => { if (event.key === "Enter") pick(); };
    body.append(tr);
  });
  if (!data.rows.length) { const tr = node("tr"); const td = node("td", "table-empty", "В текущем срезе нет объектов."); td.colSpan = 5; tr.append(td); body.append(tr); }
  $("objects-page").textContent = data.total ? `${fmt.format(data.offset + 1)}–${fmt.format(data.offset + data.rows.length)} из ${fmt.format(data.total)}` : "0 объектов";
  $("objects-prev").disabled = data.offset === 0;
  $("objects-next").disabled = data.offset + data.rows.length >= data.total;
  updateSelectionBadge();
}

function renderClusters() {
  const container = $("clusters-content"); container.replaceChildren();
  if (!S.info?.summaries.length) { container.append(node("p", "panel-empty", "Запустите кластеризацию в настройках слева.")); return; }
  S.info.summaries.forEach((summary) => {
    const row = node("div", "cluster-summary-row");
    const title = summary.is_noise ? "Шум −1" : `Кластер ${summary.cluster_id}`;
    const label = node("strong", "cluster-label", title);
    const dot = node("i", "legend-dot"); dot.style.background = clusterColor(summary.cluster_id);
    label.prepend(dot);
    row.append(label, node("span", "", `${fmt.format(summary.size)} объектов`));
    const details = node("details");
    details.append(node("summary", "", "Примеры и метаданные"));
    const examples = node("div", "cluster-examples");
    summary.example_ids.forEach((identifier) => {
      const button = node("button", "cluster-example", identifier);
      button.onclick = () => setSelection([identifier], identifier).catch((error) => toast(error.message, true));
      examples.append(button);
    });
    details.append(examples, node("pre", "cluster-meta", JSON.stringify(summary.metadata, null, 2)));
    if (summary.metadata_fields_omitted) details.append(node("p", "hint", `Ещё полей: ${summary.metadata_fields_omitted}.`));
    row.append(details); container.append(row);
  });
}

function renderTuning() {
  const report = S.info?.clustering?.tuning;
  $("tuning-result").hidden = !report;
  $("tuning-tab").hidden = !report;
  const body = $("tuning-body");
  body.replaceChildren();
  if (!report) {
    if (S.drawer === "tuning") {
      S.drawer = null;
      $("drawer-content").hidden = true;
      $("drawer-close").hidden = true;
      $("tuning-panel").hidden = true;
      $("tuning-tab").classList.remove("active");
      resizePlot();
    }
    return;
  }
  const objective = TUNING_OBJECTIVES[report.objective] || report.objective;
  $("tuning-best").textContent = `${objective}: ${compactNumber(report.best_score)} · вариант ${report.best_trial}`;
  $("show-tuning").textContent = `Варианты (${report.completed ?? report.trials.length})`;
  $("tuning-count").textContent = report.completed ?? report.trials.length;
  const scoreSpace = typeof report.score_space === "string" ? report.score_space : JSON.stringify(report.score_space);
  $("tuning-context").textContent = `${objective} · оценено объектов: ${fmt.format(report.sample_count)} · ${scoreSpace || S.info.clustering.space} · L2 · проверено ${report.completed ?? report.trials.length} из ${report.total ?? report.budget} вариантов`;
  for (const trial of report.trials) {
    const winner = trial.trial === report.best_trial && trial.status === "ok";
    const row = node("tr", winner ? "tuning-winner" : "");
    const number = node("td", "", `${trial.trial}${winner ? " · лучший" : ""}`);
    const score = node("td", "tuning-score", compactNumber(trial.score));
    const parameters = node("td", "tuning-params");
    const values = trial.params || trial.parameters || {};
    parameters.textContent = Object.entries(values).map(([key, value]) => `${key}: ${typeof value === "object" ? JSON.stringify(value) : value}`).join(" · ") || "—";
    if (trial.parameters) parameters.title = JSON.stringify(trial.parameters, null, 2);
    const status = node("td", "tuning-status", trial.status === "ok" ? `${trial.actual_clusters} кластеров` : "Ошибка");
    if (trial.error) status.append(node("p", "tuning-error", trial.error));
    for (const warning of trial.warnings || []) status.append(node("p", "hint", warning));
    row.append(number, score, parameters, status, node("td", "", Number.isFinite(trial.elapsed_seconds) ? `${compactNumber(trial.elapsed_seconds)} с` : "—"));
    body.append(row);
  }
}

function tuningToManual() {
  const clustering = S.info?.clustering;
  if (!clustering?.tuning) return;
  applyClusterControls(clustering);
  $("cluster-mode").value = "manual";
  updateParameterVisibility();
  toast("Параметры лучшего варианта перенесены. Запуск — по кнопке «Найти кластеры».");
}

function captureView() {
  return {axes: [S.view.x, S.view.y], color: S.view.color, viewport: S.view.viewport, mode: S.view.mode, pointSize: S.view.pointSize, theme: S.view.theme,
    neighborMetric: $("neighbor-metric").value, neighborScope: $("neighbor-scope").value, k: numericInput("neighbor-k"),
    controls: Object.fromEntries(CONTROL_IDS.map((id) => [id, $(id).value])), drawer: S.drawer,
    sidebarOpen: !$("app").classList.contains("sidebar-closed"), inspectorOpen: !$("app").classList.contains("inspector-closed")};
}

async function openStudy(file) {
  if (!file) return;
  setBusy(true, "Открытие исследования…");
  S.replacingDataset = true;
  try {
    await S.selectionQueue.catch(() => {});
    S.selectionQueue = Promise.resolve();
    const info = await mutation("/api/study/open", await file.arrayBuffer(), {raw: true});
    updateInfo(info, {reset: true, restore: true, filters: true});
    await refreshAll({restore: true});
    const view = info.view || {};
    if (typeof view.sidebarOpen === "boolean") $("app").classList.toggle("sidebar-closed", !view.sidebarOpen);
    if (typeof view.inspectorOpen === "boolean") $("app").classList.toggle("inspector-closed", !view.inspectorOpen);
    syncPanelAria();
    if (["objects", "clusters", ...(info.clustering?.tuning ? ["tuning"] : [])].includes(view.drawer) && S.drawer !== view.drawer) await toggleDrawer(view.drawer);
    else if (!view.drawer && S.drawer) await toggleDrawer(S.drawer);
    resizePlot();
    toast("Исследование открыто. Координаты восстановлены без пересчёта.");
  } finally { S.replacingDataset = false; setBusy(false); $("study-file").value = ""; }
}

function smallScreen() { return window.matchMedia("(max-width: 950px)").matches; }

function syncPanelAria() {
  for (const panel of ["sidebar", "inspector"]) {
    const visible = smallScreen() ? $("app").classList.contains(`${panel}-mobile-open`) : !$("app").classList.contains(`${panel}-closed`);
    $(`toggle-${panel}`).setAttribute("aria-expanded", String(visible));
  }
}

function closeMobilePanels() {
  $("app").classList.remove("sidebar-mobile-open", "inspector-mobile-open");
  $("panel-backdrop").hidden = true;
  syncPanelAria();
}

function togglePanel(panel) {
  if (smallScreen()) {
    const open = $("app").classList.contains(`${panel}-mobile-open`);
    closeMobilePanels();
    if (!open) { $("app").classList.add(`${panel}-mobile-open`); $("panel-backdrop").hidden = false; }
    $(`toggle-${panel}`).setAttribute("aria-expanded", String(!open));
  } else {
    $("app").classList.toggle(`${panel}-closed`);
    $(`toggle-${panel}`).setAttribute("aria-expanded", String(!$("app").classList.contains(`${panel}-closed`)));
    resizePlot();
  }
}

let resizeFrame;
function resizePlot() {
  cancelAnimationFrame(resizeFrame);
  resizeFrame = requestAnimationFrame(() => { if (S.plotReady) Plotly.Plots.resize($("plot")); });
}

function bindEvents() {
  on("load-dataset", "click", () => loadDataset($("dataset-path").value));
  on("dataset-path", "keydown", (event) => { if (event.key === "Enter") return loadDataset($("dataset-path").value); });
  on("dataset-list", "change", () => { $("dataset-path").value = $("dataset-list").value; });
  on("projection-form", "submit", (event) => { event.preventDefault(); return startOperation(projectionRequest()); });
  on("cluster-form", "submit", (event) => { event.preventDefault(); return startOperation(clusterRequest()); });
  ["projection-method", "cluster-method", "cluster-space", "cluster-linkage", "cluster-mode"].forEach((id) => on(id, "change", updateParameterVisibility));
  on("cancel-job", "click", cancelJob);
  on("tuning-to-manual", "click", tuningToManual);
  on("show-tuning", "click", () => { if (S.drawer !== "tuning") return toggleDrawer("tuning"); });
  ["projection-scope", "cluster-scope"].forEach((id) => on(id, "change", () => updateBounds()));
  on("normalize", "change", async () => {
    try { updateInfo(await mutation("/api/normalize", {value: $("normalize").checked})); await refreshAll(); }
    catch (error) { $("normalize").checked = S.info.normalized; throw error; }
  });
  on("clear-clusters", "click", async () => { updateInfo(await mutation("/api/clear-clusters", {})); await refreshAll(); });
  on("add-filter", "click", () => { S.filters.push(newFilter(S.info.fields[0])); renderFilters(); });
  on("apply-filters", "click", () => applyFilters(S.filters));
  on("reset-filters", "click", () => applyFilters([]));
  ["axis-x", "axis-y"].forEach((id) => on(id, "change", async () => { S.view.x = Number($("axis-x").value); S.view.y = Number($("axis-y").value); await refreshMap(); }));
  on("color-field", "change", async () => { S.view.color = $("color-field").value; await refreshMap(); });
  on("toggle-theme", "click", toggleTheme);
  on("point-size", "input", async () => {
    S.view.pointSize = Math.max(3, Math.min(16, Number($("point-size").value) || 8));
    updatePointSizeControl();
    savePreference("visualizer.pointSize", String(S.view.pointSize));
    await highlightSelection();
  });
  document.querySelectorAll("[data-mode]").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode).catch((error) => toast(error.message, true))));
  on("focus-selection", "click", focusPrimary);
  on("reset-view", "click", resetViewport);
  on("selection-count", "click", () => setSelection([]));
  ["neighbor-metric", "neighbor-scope"].forEach((id) => on(id, "change", refreshObject));
  on("neighbor-k", "change", () => { $("neighbor-k").value = Math.max(1, Math.min(1000, Math.trunc(Number($("neighbor-k").value)) || 10)); return refreshObject(); });
  on("copy-id", "click", async () => { if (S.info?.primary) { await navigator.clipboard.writeText(S.info.primary); toast("ID скопирован."); } });
  let searchTimer;
  on("search", "input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => searchObjects().catch((error) => toast(error.message, true)), 160); });
  on("search", "focus", searchObjects);
  $("search").addEventListener("keydown", searchKeyboard);
  document.querySelectorAll("[data-drawer]").forEach((button) => button.addEventListener("click", () => toggleDrawer(button.dataset.drawer).catch((error) => toast(error.message, true))));
  on("drawer-close", "click", () => toggleDrawer(S.drawer));
  on("objects-prev", "click", () => { S.objectOffset = Math.max(0, S.objectOffset - 100); return refreshObjects(); });
  on("objects-next", "click", () => { S.objectOffset += 100; return refreshObjects(); });
  on("save-study", "click", async () => { await S.selectionQueue; await S.mutationQueue; return download("/api/study/save", {view: captureView()}, "embedding-study.npz"); });
  on("open-study", "click", () => $("study-file").click());
  on("study-file", "change", () => openStudy($("study-file").files[0]));
  on("export-toggle", "click", () => { $("export-menu").hidden = !$("export-menu").hidden; $("export-toggle").setAttribute("aria-expanded", String(!$("export-menu").hidden)); });
  on("export-download", "click", async () => { await S.selectionQueue; await S.mutationQueue; await download("/api/export", {scope: $("export-scope").value, format: $("export-format").value}, `embeddings.${$("export-format").value}`); $("export-menu").hidden = true; });
  on("yt-form", "submit", (event) => { event.preventDefault(); return importYT(); });
  ["sidebar", "inspector"].forEach((panel) => on(`toggle-${panel}`, "click", () => togglePanel(panel)));
  on("inspector-close", "click", () => togglePanel("inspector"));
  document.querySelectorAll("[data-close]").forEach((button) => button.onclick = closeMobilePanels);
  on("panel-backdrop", "click", closeMobilePanels);
  on("empty-open", "click", () => { if (smallScreen() || $("app").classList.contains("sidebar-closed")) togglePanel("sidebar"); $("dataset-path").focus(); });
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".search-wrap")) $("search-results").hidden = true;
    if (!event.target.closest(".menu-wrap")) $("export-menu").hidden = true;
  });
  document.addEventListener("keydown", (event) => {
    if (event.target.closest("input,select,textarea,[contenteditable=true]")) return;
    if (event.key === "Escape") { closeMobilePanels(); $("export-menu").hidden = true; setSelection([]).catch((error) => toast(error.message, true)); }
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key.toLowerCase() === "f") focusPrimary().catch((error) => toast(error.message, true));
    if (event.key.toLowerCase() === "r") resetViewport().catch((error) => toast(error.message, true));
  });
  new ResizeObserver(resizePlot).observe(document.querySelector(".plot-container"));
  window.addEventListener("resize", () => { if (!smallScreen()) closeMobilePanels(); syncPanelAria(); resizePlot(); });
}

async function initialize() {
  applyThemeStyles();
  updatePointSizeControl();
  bindEvents();
  syncPanelAria();
  updateParameterVisibility();
  try {
    const response = await fetch("/api/config", {cache: "no-store"});
    if (!response.ok) throw new Error("Не удалось подключиться к локальному серверу.");
    S.config = await response.json();
    S.token = S.config.token;
    const result = await request("/api/workspace", {});
    S.workspace = result.workspace;
    $("dataset-path").value = S.config.default_path || "";
    const datasets = S.config.datasets || [];
    if (datasets.length) {
      $("dataset-list-label").hidden = false;
      $("dataset-list").replaceChildren(...datasets.map((entry) => option(entry.path, entry.name || entry.path)));
      if (datasets.some((entry) => entry.path === S.config.default_path)) $("dataset-list").value = S.config.default_path;
      const initial = datasets.find((entry) => entry.path === S.config.default_path) || datasets[0];
      await loadDataset(initial.path);
    }
    setupYT();
  } catch (error) { toast(error.message, true); }
}

function setupYT() {
  if (!$("yt-form")) return;
  $("yt-output").value = S.config.default_import_output || "";
  $("yt-unavailable").hidden = Boolean(S.config.yt_available);
  syncDisabled();
}

async function importYT() {
  const columns = {id: $("yt-column-id").value.trim(), embedding: $("yt-column-embedding").value.trim()};
  for (const [name, id] of [["name", "yt-column-name"], ["metadata", "yt-column-metadata"], ["embedding_size", "yt-column-size"]]) {
    if ($(id).value.trim()) columns[name] = $(id).value.trim();
  }
  const payload = {proxy: $("yt-proxy").value.trim(), table_path: $("yt-table").value.trim(), output: $("yt-output").value.trim(), columns,
    dataset_name: $("yt-name").value.trim(), space_id: $("yt-space").value.trim(), version: $("yt-version").value.trim(),
    default_metric: $("yt-metric").value, limit: numericInput("yt-limit"), offset: numericInput("yt-offset"), overwrite: false};
  if (!payload.proxy || !payload.table_path || !payload.output || !columns.id || !columns.embedding) throw new Error("Укажите прокси, таблицу, путь снимка и обязательные колонки.");
  setBusy(true, "Импорт из YT…");
  S.jobKind = "import";
  try { S.job = await mutation("/api/yt/import", payload); await pollJob(); }
  catch (error) { setBusy(false); throw error; }
}

initialize();
