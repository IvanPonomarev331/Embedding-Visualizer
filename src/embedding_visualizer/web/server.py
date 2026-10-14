"""Small localhost server. All algorithms stay in the Python compute package."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import mimetypes
import os
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

from embedding_visualizer.settings import Settings
from embedding_visualizer.workspace import Workspace, json_safe


ASSETS = Path(__file__).parent / "assets"
MAX_BODY = 512 * 1024**2


class Application:
    def __init__(self, settings):
        self.settings = settings
        self.token = secrets.token_urlsafe(32)
        self.workspaces = {}
        self.lock = threading.Lock()
        self._plotly = None

    def create_workspace(self):
        with self.lock:
            now = time.monotonic()
            for key, (workspace, used) in list(self.workspaces.items()):
                if now - used > 7200 and not (workspace.job and workspace.job["status"] == "running"):
                    del self.workspaces[key]
            if len(self.workspaces) >= 32:
                raise ValueError("Слишком много открытых сессий. Перезапустите сервер, чтобы освободить память.")
            identifier = uuid.uuid4().hex
            self.workspaces[identifier] = (Workspace(self.settings.cache_dir), now)
            return identifier

    def workspace(self, identifier):
        with self.lock:
            if identifier not in self.workspaces:
                raise ValueError("Сессия завершена. Обновите страницу и откройте сохранённое исследование.")
            workspace, _ = self.workspaces[identifier]
            self.workspaces[identifier] = (workspace, time.monotonic())
            return workspace

    def config(self):
        datasets = []
        root = self.settings.data_dir
        candidates = [root]
        if root.is_dir():
            candidates += sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)[:500]
        for path in candidates:
            if (path / "dataset.parquet").is_file() and (path / "manifest.json").is_file():
                try:
                    name = json.loads((path / "manifest.json").read_text()).get("dataset_name", path.name)
                except (ValueError, OSError):
                    name = path.name
                datasets.append({"path": str(path), "name": name})
        try:
            yt_available = importlib.util.find_spec("yt.wrapper") is not None
        except (ImportError, ModuleNotFoundError):
            yt_available = False
        return {"token": self.token, "datasets": datasets, "default_path": str(root / "example"),
                "yt_available": yt_available, "default_import_output": str(root / "yt-snapshot")}

    def plotly(self):
        if self._plotly is None:
            from plotly.offline import get_plotlyjs
            self._plotly = get_plotlyjs().encode()
        return self._plotly

    def import_yt(self, workspace, params):
        from embedding_visualizer.adapters.yt import import_yt_dataset
        with workspace.lock:
            workspace.idle()
            job = dict(id=uuid.uuid4().hex, status="running", operation="import", method="YT")
            workspace.job = job

        def run():
            try:
                path = import_yt_dataset(**params)
                with workspace.lock:
                    workspace.job["status"] = "done"
                    workspace.load(path)
                    workspace.job = dict(job, status="done", cache_hit=False)
            except Exception as exc:
                with workspace.lock:
                    workspace.job = dict(job, status="error", error=str(exc))
        threading.Thread(target=run, daemon=True).start()
        return job.copy()


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, settings):
        self.application = Application(settings)
        super().__init__((settings.address, settings.port), Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "EmbeddingVisualizer/0.2"

    def log_message(self, format, *args):
        # Avoid printing local paths, IDs and metadata in default request logs.
        pass

    def send(self, body, status=200, content_type="application/json; charset=utf-8", filename=None):
        if not isinstance(body, bytes):
            body = json.dumps(json_safe(body), ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if content_type.startswith("application/json") else "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def allowed_request(self, token_required=False):
        port = self.server.server_port
        address = self.server.application.settings.address
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"{address}:{port}"}
        host = self.headers.get("Host", "")
        if host not in hosts:
            self.send({"error": "Unrecognized local host."}, 403)
            return False
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{host}":
            self.send({"error": "Cross-origin requests are not allowed."}, 403)
            return False
        if token_required and not secrets.compare_digest(self.headers.get("X-Visualizer-Token", ""), self.server.application.token):
            self.send({"error": "Refresh the page to reconnect to the local server."}, 403)
            return False
        return True

    def do_GET(self):
        if not self.allowed_request():
            return
        url = urlsplit(self.path)
        try:
            if url.path == "/":
                self.send((ASSETS / "index.html").read_bytes(), content_type="text/html; charset=utf-8")
            elif url.path == "/health":
                self.send({"ok": True})
            elif url.path == "/assets/plotly.min.js":
                self.send(self.server.application.plotly(), content_type="text/javascript; charset=utf-8")
            elif url.path.startswith("/assets/"):
                name = url.path.removeprefix("/assets/")
                if name not in {p.name for p in ASSETS.iterdir() if p.is_file()}:
                    self.send({"error": "Not found."}, 404)
                    return
                self.send((ASSETS / name).read_bytes(), content_type=mimetypes.guess_type(name)[0] or "application/octet-stream")
            elif url.path == "/api/config":
                self.send(self.server.application.config())
            elif url.path.startswith("/api/"):
                if not self.allowed_request(token_required=True):
                    return
                workspace = self.server.application.workspace(self.headers.get("X-Visualizer-Workspace"))
                query = {key: values[-1] for key, values in parse_qs(url.query).items()}
                if url.path == "/api/state":
                    self.send(workspace.info())
                elif url.path == "/api/job":
                    with workspace.lock:
                        self.send(workspace.job or {"status": "idle"})
                elif url.path == "/api/objects":
                    self.send(workspace.objects(query.get("query", ""), int(query.get("offset", 0)), int(query.get("limit", 100))))
                elif url.path == "/api/object":
                    self.send(workspace.object_detail(query.get("id"), int(query.get("k", 10)), query.get("metric", "cosine"), query.get("scope", "all")))
                else:
                    self.send({"error": "Not found."}, 404)
            else:
                self.send({"error": "Not found."}, 404)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.send({"error": str(exc)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        if not self.allowed_request(token_required=True):
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                self.send({"error": "Размер запроса должен быть от 1 байта до 512 MiB."}, 413)
                return
            path = urlsplit(self.path).path
            body = self.rfile.read(length)
            if path == "/api/study/open":
                data = None
            else:
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("Ожидается Content-Type: application/json.")
                data = json.loads(body)
                if not isinstance(data, dict):
                    raise ValueError("Ожидается JSON-объект.")
            application = self.server.application
            if path == "/api/workspace":
                self.send({"workspace": application.create_workspace()})
                return
            workspace = application.workspace(self.headers.get("X-Visualizer-Workspace"))
            if path == "/api/load":
                result = workspace.load(data["path"])
            elif path == "/api/map":
                result = workspace.map_data(**data)
            elif path == "/api/filters":
                result = workspace.set_filters(data["filters"])
            elif path == "/api/normalize":
                result = workspace.set_normalization(data["value"])
            elif path == "/api/operation":
                result = workspace.start_operation(**data)
            elif path == "/api/job/cancel":
                result = workspace.cancel_job()
            elif path == "/api/clear-clusters":
                result = workspace.clear_clusters()
            elif path == "/api/selection":
                result = workspace.select(**data)
            elif path == "/api/study/save":
                self.send(workspace.save(data.get("view", {})), content_type="application/octet-stream", filename="research.npz")
                return
            elif path == "/api/study/open":
                result = workspace.restore(body)
            elif path == "/api/export":
                format = data.get("format", "csv")
                if format not in ("csv", "parquet"):
                    raise ValueError("Формат экспорта: csv или parquet.")
                self.send(workspace.export(data.get("scope", "selected"), format),
                    content_type="text/csv; charset=utf-8" if format == "csv" else "application/octet-stream",
                    filename=f"embeddings.{format}")
                return
            elif path == "/api/yt/import":
                result = application.import_yt(workspace, data)
            else:
                self.send({"error": "Not found."}, 404)
                return
            self.send(result)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.send({"error": str(exc)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass


def run(settings):
    os.environ["NUMBA_CACHE_DIR"] = str(settings.cache_dir / "numba")
    server = LocalServer(settings)
    print(f"Embedding Visualizer\nhttp://{settings.address}:{server.server_port}\nCtrl+C — остановить сервер", flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    return run(Settings(**vars(args)))


if __name__ == "__main__":
    raise SystemExit(main())
