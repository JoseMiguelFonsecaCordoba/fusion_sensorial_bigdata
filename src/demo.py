"""Demos para la presentación (antes vivían en el notebook).

  python -m src.demo              -> las 3 demos contrastantes con su narrativa
  python -m src.demo --preset jazz --color "#1F6966" --presupuesto 0.5
"""
import argparse

from . import recommender, sensory

DEMOS = [
    ("Vibra enérgica y vibrante (reggaeton + rojo/naranja)",
     dict(preset="reggaeton", color="#FF5733", presupuesto=0.0, solo_joyas=False)),
    ("Vibra melancólica y pacífica (música lenta + azul oscuro)",
     dict(music=(70, 0.2, 0.1, 0.25), color="#1A365D", presupuesto=0.0, solo_joyas=False)),
    ("Joyas ocultas con presupuesto bajo (verde)",
     dict(color="#2ECC71", presupuesto=0.9, solo_joyas=True)),
]


def _level(x, high, mid, low):
    return high if x > 0.65 else mid if x > 0.4 else low


def narrate(p):
    """Traduce el perfil (A, P, W) a una frase para la audiencia."""
    animo = _level(p.P, "optimismo radiante", "una estabilidad tranquila", "nostalgia reflexiva")
    energia = _level(p.A, "alta intensidad y ganas de moverte", "un ritmo dinámico pero moderado",
                     "calma y contemplación")
    calidez = _level(p.W, "cálidos y acogedores", "templados", "fríos y refrescantes")
    return (f"Buscas un lugar con {animo}, {energia}, y tonos {calidez}.\n"
            f"Activación {p.A:.2f} · Positividad {p.P:.2f} · Calidez {p.W:.2f}")


def run(units, title, preset=None, music=None, color=None, presupuesto=0.0, solo_joyas=False, top=3):
    if preset:
        m = sensory.music_axes(**sensory.load_presets()[preset])
    elif music:
        m = sensory.music_axes(*music)
    else:
        m = None
    profile = sensory.build_profile(m, sensory.color_profile(color) if color else None)
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}\n{narrate(profile)}\n")
    for i, r in enumerate(recommender.recommend(units, profile, presupuesto, top, solo_joyas=solo_joyas), 1):
        print(f"{i}. {r['town'].replace('_', ' ')} · {r['type']}  [{r['region']}]  {r['score']:.1f}/100")
        print(f"   {r['why']}\n")


def main():
    ap = argparse.ArgumentParser(description="Demos con narrativa")
    ap.add_argument("--preset")
    ap.add_argument("--color")
    ap.add_argument("--presupuesto", type=float, default=0.0)
    ap.add_argument("--solo-joyas", action="store_true")
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()
    units = recommender.load_profiles()
    if a.preset or a.color:
        if a.preset and a.preset not in sensory.load_presets():
            raise SystemExit(f"Preset desconocido: {a.preset}")
        run(units, "Demo personalizada", a.preset, None, a.color, a.presupuesto, a.solo_joyas, a.top)
        return
    for title, kw in DEMOS:
        run(units, title, **kw)


if __name__ == "__main__":
    main()
