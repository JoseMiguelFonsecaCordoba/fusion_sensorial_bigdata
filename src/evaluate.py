"""Evaluación (no existe ground truth, así que se miden coherencia y estabilidad).

  python -m src.evaluate               -> pruebas de coherencia (rápidas, sin Spark)
  python -m src.evaluate --presets     -> además, top-2 por preset de género
  python -m src.evaluate --split-half  -> además, fiabilidad por mitades con PySpark
"""
import argparse
import random
from collections import Counter

from . import config, recommender, sensory
from .lexicon import TOURISM_DIMS

SCENARIOS = {
    "fiesta (rápido, intenso, cálido)": (
        dict(bpm=150, energy=0.9, danceability=0.85, valence=0.8), "#FF3300",
        ["vida_nocturna", "aventura"]),
    "calma (lento, suave, frío)": (
        dict(bpm=70, energy=0.2, danceability=0.3, valence=0.4), "#6FA8DC",
        ["descanso", "naturaleza"]),
}


def _profile(music_kwargs, hex_code):
    return sensory.build_profile(sensory.music_axes(**music_kwargs), sensory.color_profile(hex_code))


def coherence_checks(units):
    """El top-5 debe tener, en las dimensiones esperadas, mayor puntuación que el promedio general."""
    print("== Coherencia sensorial ==")
    ok = True
    for name, (music, hex_code, expected) in SCENARIOS.items():
        top = recommender.recommend(units, _profile(music, hex_code), top=5)
        mean_top = sum(u["t"][d] for u in top for d in expected) / (5 * len(expected))
        passed = mean_top > 0.0
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  {name}: t medio en {expected} del top-5 = {mean_top:+.2f} (promedio general = 0)")
        for u in top[:3]:
            print(f"      - {u['town'].replace('_', ' ')} · {u['type']} ({u['score']:.1f})")

    print("\n== Efecto del presupuesto ==")
    prof = _profile(*SCENARIOS["calma (lento, suave, frío)"][:2])
    v0 = [u["v_norm"] for u in recommender.recommend(units, prof, 0.0, top=10)]
    v1 = [u["v_norm"] for u in recommender.recommend(units, prof, 1.0, top=10)]
    passed = sum(v1) / 10 > sum(v0) / 10
    ok &= passed
    print(f"{'PASS' if passed else 'FAIL'}  valor medio del top-10: presupuesto 0 = {sum(v0)/10:.2f}  ->  presupuesto 1 = {sum(v1)/10:.2f}")

    print("\n== Diversidad ==")
    seen = set()
    for music, hex_code, _ in SCENARIOS.values():
        seen |= {(u["town"], u["type"]) for u in recommender.recommend(units, _profile(music, hex_code), top=10)}
    print(f"Unidades distintas en los top-10 de {len(SCENARIOS)} perfiles: {len(seen)} de {len(units)}")
    coverage(units)
    return ok


def coverage(units, n=200, seed=7):
    """Con n perfiles sensoriales aleatorios: qué fracción del catálogo llega a algún top-10."""
    rng = random.Random(seed)
    seen, by_type = set(), Counter()
    for _ in range(n):
        music = sensory.music_axes(rng.uniform(60, 180), rng.random(), rng.random(), rng.random())
        color = sensory.color_profile("#%02X%02X%02X" % tuple(rng.randrange(256) for _ in range(3)))
        for u in recommender.recommend(units, sensory.build_profile(music, color), rng.random(), top=10):
            seen.add((u["town"], u["type"]))
            by_type[u["type"]] += 1
    total = sum(by_type.values())
    print(f"Cobertura con {n} perfiles aleatorios: {len(seen)}/{len(units)} unidades ({len(seen)/len(units):.0%})")
    print("  Reparto del top-10 por Type: " + ", ".join(f"{t} {c/total:.0%}" for t, c in by_type.most_common()))


def presets_table(units):
    print("\n== Top-2 por preset (solo música) ==")
    for name, v in sensory.load_presets().items():
        top = recommender.recommend(units, sensory.build_profile(sensory.music_axes(**v)), top=2)
        print(f"  {name:<12} " + " | ".join(f"{u['town'].replace('_', ' ')} · {u['type']} ({u['score']:.0f})" for u in top))


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks, i = [0.0] * len(values), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx, vy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return cov / (vx * vy) ** 0.5 if vx and vy else float("nan")


def split_half(units):
    """Divide las reseñas al azar en dos mitades y correlaciona (Spearman) las tasas por dimensión."""
    from pyspark.sql import SparkSession, functions as F
    from .build_profiles import load_reviews

    spark = SparkSession.builder.master("local[*]").appName("SplitHalf").config("spark.driver.memory", "2g").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    df = load_reviews(spark).withColumn("half", (F.rand(seed=42) < 0.5).cast("int"))
    rates = df.groupBy("Town", "Type", "half").agg(*[F.avg(d).alias(d) for d in TOURISM_DIMS]).collect()
    spark.stop()

    eligible = {(u["town"], u["type"]) for u in units}
    by_unit = {}
    for r in rates:
        if (r["Town"], r["Type"]) in eligible:
            by_unit.setdefault((r["Town"], r["Type"]), {})[r["half"]] = r
    pairs = [v for v in by_unit.values() if 0 in v and 1 in v]
    print(f"\n== Fiabilidad por mitades (Spearman, {len(pairs)} unidades) ==")
    for d in TOURISM_DIMS:
        print(f"  {d:<14} rho = {spearman([p[0][d] for p in pairs], [p[1][d] for p in pairs]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-half", action="store_true")
    ap.add_argument("--presets", action="store_true")
    args = ap.parse_args()
    units = recommender.load_profiles(config.PROFILES_PATH)
    ok = coherence_checks(units)
    if args.presets:
        presets_table(units)
    if args.split_half:
        split_half(units)
    print("\nResultado global de coherencia:", "OK" if ok else "REVISAR")


if __name__ == "__main__":
    main()
