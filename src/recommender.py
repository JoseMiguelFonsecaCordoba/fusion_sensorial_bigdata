"""Motor de recomendación.

1) Perfil sensorial (A,P,W) -> vector de intención turística tau = M·(a,p,w) con ejes centrados.
2) Afinidad = (coseno(tau, t) + 1) / 2, donde t es el perfil del destino (centrado, por Type).
3) Calidad-valor = (1 - presupuesto)*q_norm + presupuesto*v_norm.
4) Score = 100 * afinidad^BETA * calidad_valor^(1-BETA).
"""
import json
import math

from . import config
from .lexicon import LABELS, M, TOURISM_DIMS


def load_profiles(path=config.PROFILES_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Genera los perfiles primero con: python -m src.build_profiles")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["units"]


def intent_vector(profile):
    x = (2 * profile.A - 1, 2 * profile.P - 1, 2 * profile.W - 1)
    return {d: sum(w * xi for w, xi in zip(M[d], x)) for d in TOURISM_DIMS}


def _cosine(tau, t):
    dot = sum(tau[d] * t[d] for d in TOURISM_DIMS)
    nt = math.sqrt(sum(v * v for v in tau.values()))
    nu = math.sqrt(sum(t[d] ** 2 for d in TOURISM_DIMS))
    return dot / (nt * nu) if nt > 0 and nu > 0 else 0.0


def _explain(unit, tau, presupuesto):
    # Solo cuenta como "encaje" lo que el usuario busca (tau>0) y el destino tiene por encima del promedio (t>0).
    pos = {d: tau[d] * unit["t"][d] for d in TOURISM_DIMS if tau[d] > 0 and unit["t"][d] > 0}
    parts = []
    if pos:
        total = sum(pos.values())
        top = sorted(pos, key=pos.get, reverse=True)[:2]
        parts.append("encaja con tu perfil en " + " y ".join(
            f"{LABELS[d]} ({pos[d] / total:.0%} de la afinidad)" for d in top))
    else:
        parts.append("no sobresale en lo que buscas; su puntaje viene de evitar lo que no buscas y de su calidad/valor")
    if presupuesto >= 0.3:
        parts.append(f"{unit['rate_barato']:.1%} de sus reseñas mencionan precios bajos "
                     f"frente a {unit['rate_caro']:.1%} que mencionan precios altos")
    parts.append(f"polaridad media {unit['pol_mean']:.2f} en {unit['n']:,} reseñas")
    if unit["joya"]:
        parts.append("joya oculta: buen valor y calidad con poca popularidad")
    if unit["top_terms"]:
        parts.append("términos distintivos: " + ", ".join(unit["top_terms"][:5]))
    return "; ".join(parts)


def score_units(units, profile, presupuesto=0.0, beta=config.BETA):
    tau = intent_vector(profile)
    neutral = all(abs(v) < 1e-9 for v in tau.values())
    scored = []
    for u in units:
        affinity = 0.5 if neutral else (_cosine(tau, u["t"]) + 1) / 2
        qv = max((1 - presupuesto) * u["q_norm"] + presupuesto * u["v_norm"], 0.01)
        scored.append({**u, "affinity": affinity, "quality_value": qv,
                       "score": 100 * affinity ** beta * qv ** (1 - beta),
                       "why": _explain(u, tau, presupuesto)})
    return sorted(scored, key=lambda r: r["score"], reverse=True)


def recommend(units, profile, presupuesto=0.0, top=5, tipo=None, vista="unidades", solo_joyas=False):
    pool = [u for u in units if (tipo is None or u["type"].lower() == tipo.lower())
            and (not solo_joyas or u["joya"])]
    ranked = score_units(pool, profile, presupuesto)
    if vista == "unidades":
        return ranked[:top]
    towns = {}
    for r in ranked:
        towns.setdefault(r["town"], []).append(r)
    result = []
    for town, rows in towns.items():
        best = rows[0]
        result.append({**best, "score": sum(r["score"] for r in rows) / len(rows),
                       "types": ", ".join(f"{r['type']} ({r['score']:.0f})" for r in rows)})
    return sorted(result, key=lambda r: r["score"], reverse=True)[:top]
