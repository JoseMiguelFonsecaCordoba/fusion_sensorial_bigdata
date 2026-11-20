"""Front web: servidor HTTP (solo librería estándar) que expone el recomendador real.

  python -m src.web            -> http://localhost:8000
  python -m src.web --port 9000
"""
import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config, recommender, sensory
from .lexicon import LABELS, TOURISM_DIMS

INDEX = config.ROOT / "web" / "index.html"
MAX_BODY = 8 * 1024 * 1024  # imágenes en base64
TYPES = ("Hotel", "Restaurant", "Attractive")


def _num(d, key, lo, hi, default=None):
    v = d.get(key, default)
    if v is None:
        return None
    v = float(v)
    if not lo <= v <= hi:
        raise ValueError(f"{key} debe estar entre {lo} y {hi}")
    return v


def recommend(req, units, presets):
    music = None
    if req.get("preset"):
        if req["preset"] not in presets:
            raise ValueError(f"Preset desconocido: {req['preset']}")
        music = sensory.music_axes(**presets[req["preset"]])
    elif req.get("music"):
        m = req["music"]
        music = sensory.music_axes(_num(m, "bpm", 40, 220), _num(m, "energy", 0, 1),
                                   _num(m, "danceability", 0, 1), _num(m, "valence", 0, 1))

    color = None
    if req.get("image"):
        raw = base64.b64decode(req["image"].split(",", 1)[-1])
        color = sensory.color_profile(image_path=io.BytesIO(raw))
    elif req.get("color"):
        color = sensory.color_profile(hex_code=str(req["color"]))
    if music is None and color is None:
        raise ValueError("Elige al menos música o color")

    tipo = req.get("tipo") or None
    if tipo is not None and tipo not in TYPES:
        raise ValueError(f"tipo inválido: {tipo}")
    vista = "destinos" if req.get("vista") == "destinos" else "unidades"
    top = int(_num(req, "top", 1, 30, 6))

    profile = sensory.build_profile(music, color, _num(req, "alpha", 0, 1, config.FUSION_ALPHA))
    tau = recommender.intent_vector(profile)
    results = recommender.recommend(units, profile, _num(req, "presupuesto", 0, 1, 0.0), top,
                                    tipo, vista, bool(req.get("solo_joyas")))
    return {
        "profile": {"A": profile.A, "P": profile.P, "W": profile.W, "sources": profile.sources,
                    "palette": (color or {}).get("palette", [])},
        "tau": tau,
        "results": results,
    }


def make_handler(units, presets):
    meta = json.dumps({"presets": presets, "dims": TOURISM_DIMS, "labels": LABELS,
                       "types": TYPES, "n_units": len(units)}).encode()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/meta":
                self._send(200, meta)
            else:
                self._send(404, b'{"error": "no encontrado"}')

        def do_POST(self):
            if self.path != "/api/recommend":
                return self._send(404, b'{"error": "no encontrado"}')
            size = int(self.headers.get("Content-Length") or 0)
            if size > MAX_BODY:
                return self._send(413, b'{"error": "la imagen es demasiado grande (max. ~6 MB)"}')
            try:
                out = recommend(json.loads(self.rfile.read(size)), units, presets)
                self._send(200, json.dumps(out, ensure_ascii=False).encode())
            except Exception as e:  # entrada inválida del cliente: se reporta, no tumba el servidor
                self._send(400, json.dumps({"error": str(e)}, ensure_ascii=False).encode())

        def log_message(self, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    units, presets = recommender.load_profiles(), sensory.load_presets()
    server = ThreadingHTTPServer((a.host, a.port), make_handler(units, presets))
    print(f"Fusión Sensorial lista en http://{a.host}:{a.port}  (Ctrl+C para salir)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
