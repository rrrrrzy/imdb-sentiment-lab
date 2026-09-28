"""Loopback-only teaching demo with read-only report and inference endpoint."""
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from .config import Paths
from .predict import load_selected_model, predict_review


def serve(paths: Paths, port: int = 8765) -> None:
    name, model = load_selected_model(paths)

    class Handler(SimpleHTTPRequestHandler):
        def respond(self, status: int, payload: dict):
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self):
            if self.path != "/api/predict":
                self.respond(404, {"error": "Unknown endpoint"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 50000:
                    raise ValueError("Invalid request size")
                origin = self.headers.get("Origin")
                if origin and origin not in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}:
                    raise ValueError("Only local report requests are allowed")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object")
                self.respond(200, predict_review(name, model, payload.get("text")))
            except (ValueError, UnicodeDecodeError) as error:
                self.respond(400, {"error": str(error)})

    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(paths.reports)))
    print(f"Report and prediction demo: http://127.0.0.1:{port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
