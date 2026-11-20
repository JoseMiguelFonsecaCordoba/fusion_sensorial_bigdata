"""CLI de Fusión Sensorial.

Ejemplos:
  python -m src.main --preset reggaeton --color "#FF8800" --presupuesto 0.5
  python -m src.main --listar-presets
  python -m src.main --bpm 150 --energy 0.9 --danceability 0.85 --valence 0.8 --color "#FF3300"
  python -m src.main --color "#6FA8DC" --presupuesto 0.8 --tipo Hotel --top 5
  python -m src.main --playlist-csv mi_playlist.csv --imagen foto.jpg --vista destinos
"""
import argparse

from . import config, recommender, sensory


def parse_args():
    p = argparse.ArgumentParser(description="Recomendador turístico por fusión sensorial (música + color)")
    m = p.add_argument_group("música (preset, manual o CSV)")
    m.add_argument("--preset", help="género/ambiente de data/presets.json (ver --listar-presets)")
    m.add_argument("--listar-presets", action="store_true")
    m.add_argument("--bpm", type=float)
    m.add_argument("--energy", type=float)
    m.add_argument("--danceability", type=float)
    m.add_argument("--valence", type=float)
    m.add_argument("--playlist-csv", help="CSV con columnas bpm|tempo, energy, danceability, valence")
    c = p.add_argument_group("color")
    c.add_argument("--color", help="código hexadecimal, p. ej. #FF5500")
    c.add_argument("--imagen", help="ruta de una imagen (requiere Pillow)")
    p.add_argument("--presupuesto", type=float, default=0.0,
                   help="0 = el precio no importa ... 1 = máxima importancia al valor/precio bajo")
    p.add_argument("--tipo", choices=["Hotel", "Restaurant", "Attractive"])
    p.add_argument("--vista", choices=["unidades", "destinos"], default="unidades")
    p.add_argument("--solo-joyas", action="store_true", help="solo 'joyas ocultas'")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--alpha", type=float, default=config.FUSION_ALPHA, help="peso música vs color")
    return p.parse_args()


def music_from_args(a):
    manual = [a.bpm, a.energy, a.danceability, a.valence]
    if a.playlist_csv:
        avg = sensory.load_music_csv(a.playlist_csv)
        print(f"Playlist CSV: {avg['tracks']} canciones (promedio bpm={avg['bpm']:.0f}, "
              f"energy={avg['energy']:.2f}, dance={avg['danceability']:.2f}, valence={avg['valence']:.2f})")
        return sensory.music_axes(avg["bpm"], avg["energy"], avg["danceability"], avg["valence"])
    if a.preset:
        presets = sensory.load_presets()
        if a.preset.lower() not in presets:
            raise SystemExit(f"Preset desconocido: {a.preset}. Opciones: {', '.join(presets)}")
        return sensory.music_axes(**presets[a.preset.lower()])
    if any(v is not None for v in manual):
        if any(v is None for v in manual):
            raise SystemExit("Para música manual indica --bpm, --energy, --danceability y --valence")
        return sensory.music_axes(*manual)
    return None


def main():
    a = parse_args()
    if a.listar_presets:
        for name, v in sensory.load_presets().items():
            print(f"{name:<12} bpm={v['bpm']:<4} energy={v['energy']:.2f}  "
                  f"dance={v['danceability']:.2f}  valence={v['valence']:.2f}")
        return
    if not 0.0 <= a.presupuesto <= 1.0:
        raise SystemExit("--presupuesto debe estar entre 0 y 1")
    if not 0.0 <= a.alpha <= 1.0:
        raise SystemExit("--alpha debe estar entre 0 y 1")
    music = music_from_args(a)
    color = sensory.color_profile(a.color, a.imagen) if (a.color or a.imagen) else None
    if music is None and color is None:
        raise SystemExit("Indica al menos una modalidad: música (manual/CSV) o color (--color/--imagen)")

    profile = sensory.build_profile(music, color, a.alpha)
    print(f"\nPerfil sensorial ({' + '.join(profile.sources)}): "
          f"activación={profile.A:.2f}  positividad={profile.P:.2f}  calidez={profile.W:.2f}")
    if color and color.get("palette"):
        print("Paleta:", " ".join(color["palette"]))

    units = recommender.load_profiles()
    results = recommender.recommend(units, profile, a.presupuesto, a.top, a.tipo, a.vista, a.solo_joyas)
    print(f"\nTop {len(results)} ({a.vista}, presupuesto={a.presupuesto}):\n")
    for i, r in enumerate(results, 1):
        label = r["town"] if a.vista == "destinos" else f"{r['town']} · {r['type']}"
        print(f"{i}. {label.replace('_', ' ')}  [{r['region']}]  compatibilidad {r['score']:.1f}/100")
        if a.vista == "destinos":
            print(f"   Tipos: {r['types']}")
        print(f"   {r['why']}\n")


if __name__ == "__main__":
    main()
