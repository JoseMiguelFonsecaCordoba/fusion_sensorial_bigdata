"""Módulo sensorial: música (entrada manual o CSV) + color (hex o imagen) -> perfil (A, P, W).

Ejes, todos en [0, 1]:
  A = activación (ritmo/excitación)   P = positividad (tono afectivo)   W = calidez (cálido vs frío)

Nota: no se usa la API de Spotify (Audio Features está cerrado para apps nuevas y la
Developer Policy prohíbe construir perfiles de usuario con su contenido). La música llega
por entrada manual o por un CSV con columnas bpm|tempo, energy, danceability, valence.
Los coeficientes son heurísticos (asociaciones cross-modales), no estimados con datos.
"""
import colorsys
import csv
import json
import math
from dataclasses import dataclass, field

BPM_MIN, BPM_MAX = 60.0, 180.0


def _clip01(x):
    return max(0.0, min(1.0, x))


@dataclass
class SensoryProfile:
    A: float
    P: float
    W: float
    sources: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------- música
def music_axes(bpm, energy, danceability, valence):
    for name, v in (("energy", energy), ("danceability", danceability), ("valence", valence)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} debe estar entre 0 y 1 (recibido {v})")
    bpm_n = _clip01((bpm - BPM_MIN) / (BPM_MAX - BPM_MIN))
    return {"A": 0.35 * bpm_n + 0.35 * energy + 0.30 * danceability, "P": valence}


def load_presets(path=None):
    """Presets de género/ambiente (data/presets.json). Las claves que empiezan con '_' son notas."""
    from . import config
    with open(path or config.PRESETS_PATH, encoding="utf-8") as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def load_music_csv(path):
    """Promedia las canciones de un CSV (bpm|tempo, energy, danceability, valence)."""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r = {k.strip().lower(): v for k, v in r.items()}
            try:
                rows.append((float(r.get("bpm", r.get("tempo"))), float(r["energy"]),
                             float(r["danceability"]), float(r["valence"])))
            except (TypeError, ValueError, KeyError):
                continue
    if not rows:
        raise ValueError("El CSV no contiene filas válidas con bpm|tempo, energy, danceability, valence")
    means = [sum(c) / len(rows) for c in zip(*rows)]
    return {"bpm": means[0], "energy": means[1], "danceability": means[2],
            "valence": means[3], "tracks": len(rows)}


# ---------------------------------------------------------------- color
def parse_hex(code):
    code = code.strip().lstrip("#")
    if len(code) == 3:
        code = "".join(c * 2 for c in code)
    if len(code) != 6:
        raise ValueError(f"Código hexadecimal inválido: {code!r}")
    return tuple(int(code[i:i + 2], 16) for i in (0, 2, 4))


def color_axes(rgb):
    """RGB -> HSV -> (A, P, W). La calidez usa el hue (máx. en naranja ~30°, mín. en azul ~210°)."""
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    warmth = (1 + math.cos(math.radians(h * 360 - 30))) / 2
    # Un gris/blanco/negro no tiene hue: la calidez se acerca a neutra (0.5) si hay poca saturación.
    warmth = 0.5 + (warmth - 0.5) * min(1.0, s / 0.25)
    return {"A": 0.5 * s + 0.3 * warmth + 0.2 * v, "P": 0.5 * v + 0.5 * warmth, "W": warmth,
            "hsv": (round(h * 360), round(s, 2), round(v, 2))}


def palette_from_image(path, k=5):
    """Paleta dominante (K colores por cuantización) con la proporción de cada color."""
    from PIL import Image  # dependencia opcional: solo si se carga una imagen
    img = Image.open(path).convert("RGB")
    img.thumbnail((128, 128))
    q = img.quantize(colors=k)
    pal, counts = q.getpalette(), q.getcolors()
    total = sum(c for c, _ in counts)
    return [((pal[3 * i], pal[3 * i + 1], pal[3 * i + 2]), c / total) for c, i in counts]


def color_profile(hex_code=None, image_path=None):
    """Para imágenes se promedian los EJES por proporción de color (promediar hues sería incorrecto)."""
    if image_path:
        palette = palette_from_image(image_path)
    elif hex_code:
        palette = [(parse_hex(hex_code), 1.0)]
    else:
        raise ValueError("Se requiere un hex o una imagen")
    axes = [(color_axes(rgb), w) for rgb, w in palette]
    out = {k: sum(a[k] * w for a, w in axes) for k in ("A", "P", "W")}
    out["palette"] = ["#%02X%02X%02X" % rgb for rgb, _ in palette]
    return out


# ---------------------------------------------------------------- fusión
def build_profile(music=None, color=None, alpha=0.5):
    """Fusión tardía: A y P = alpha*música + (1-alpha)*color; W solo viene del color (neutra 0.5 si falta)."""
    if music is None and color is None:
        raise ValueError("Se necesita al menos una modalidad (música o color)")
    if music and color:
        A = alpha * music["A"] + (1 - alpha) * color["A"]
        P = alpha * music["P"] + (1 - alpha) * color["P"]
    else:
        src = music or color
        A, P = src["A"], src["P"]
    W = color["W"] if color else 0.5
    sources = (["música"] if music else []) + (["color"] if color else [])
    return SensoryProfile(_clip01(A), _clip01(P), _clip01(W), sources,
                          {"music": music, "color": color})
