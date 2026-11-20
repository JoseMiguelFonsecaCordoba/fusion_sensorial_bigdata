"""Procesamiento Big Data con PySpark: reseñas -> perfiles por (Town, Type).

Pipeline (todo distribuido en DataFrames):
  1. Lectura del CSV y normalización de texto (sin acentos, solo letras).
  2. Marcas binarias por dimensión (¿la reseña menciona la dimensión?).
  3. Agregación por (Town, Region, Type): n, polaridad media, tasas de mención.
  4. Calidad suavizada (bayesiana) e índice de valor V = log((barato+a)/(caro+a)).
  5. Percentiles (Window) DENTRO de cada Type y GLOBALES; el perfil turístico mezcla ambos.
     El del tipo quita el sesgo (un Restaurant siempre menciona más gastronomía que un Hotel);
     el global conserva la magnitud real entre tipos (si no, los Restaurant dominan el top).
  6. TF-IDF a nivel de destino (Town) para extraer los términos distintivos.
  7. Escritura de data/processed/profiles.json (tabla pequeña, <=120 filas).

Ejecución:  python -m src.build_profiles
"""
import json

from pyspark.sql import SparkSession, Window, functions as F

from . import config
from .lexicon import DIMS, TOURISM_DIMS

ACCENTS_FROM, ACCENTS_TO = "áéíóúüñ", "aeiouun"


def normalize_text(df):
    """Title + Review en minúsculas, sin acentos y solo letras (concat_ws ignora nulos)."""
    txt = F.lower(F.concat_ws(" ", F.col("Title"), F.col("Review")))
    txt = F.translate(txt, ACCENTS_FROM, ACCENTS_TO)
    txt = F.regexp_replace(txt, r"[^a-z\s]", " ")
    txt = F.regexp_replace(txt, r"\s+", " ")
    return df.withColumn("text", F.trim(txt))


def add_dimension_flags(df):
    """Una columna 0/1 por dimensión: la reseña la menciona o no."""
    for name, alternatives in DIMS.items():
        df = df.withColumn(name, F.col("text").rlike(rf"\b(?:{alternatives})\b").cast("int"))
    return df


def load_reviews(spark):
    df = (
        # file:// explícito: si hay Hadoop instalado, una ruta sin esquema se busca en HDFS
        spark.read.csv(config.DATA_PATH.resolve().as_uri(), header=True, inferSchema=True,
                       multiLine=True, escape='"')
        .withColumn("Polarity", F.col("Polarity").cast("double"))
        .dropna(subset=["Review", "Polarity", "Town", "Region", "Type"])
    )
    return add_dimension_flags(normalize_text(df))


def percentile_within_type(df, source_col, target_col):
    """percent_rank en [0,1] dentro de cada Type (0 = peor del tipo, 1 = mejor)."""
    w = Window.partitionBy("Type").orderBy(F.col(source_col).asc())
    return df.withColumn(target_col, F.percent_rank().over(w))


def percentile_global(df, source_col, target_col):
    """percent_rank en [0,1] sobre todas las unidades (una sola partición: la tabla es pequeña)."""
    return df.withColumn(target_col, F.percent_rank().over(Window.orderBy(F.col(source_col).asc())))


def build_units(df):
    """Tabla (Town, Region, Type) con calidad, valor, perfil turístico y popularidad."""
    mu = df.agg(F.avg("Polarity")).first()[0]

    aggs = [F.count("*").alias("n"), F.avg("Polarity").alias("pol_mean")]
    aggs += [F.avg(d).alias(f"rate_{d}") for d in DIMS]
    units = (
        df.groupBy("Town", "Region", "Type").agg(*aggs)
        .filter(F.col("n") >= config.MIN_REVIEWS)
    )

    m = config.PRIOR_WEIGHT
    units = units.withColumn(
        "quality_shrunk", (F.col("n") * F.col("pol_mean") + F.lit(m * mu)) / (F.col("n") + F.lit(m))
    )
    a = config.VALUE_ALPHA
    units = units.withColumn(
        "value_raw",
        F.log((F.col("rate_valor_barato") + F.lit(a)) / (F.col("rate_valor_caro") + F.lit(a))),
    )

    for d in TOURISM_DIMS:
        units = percentile_within_type(units, f"rate_{d}", f"pctT_{d}")
        units = percentile_global(units, f"rate_{d}", f"pctG_{d}")
        w = config.TYPE_PCT_WEIGHT
        units = units.withColumn(f"pct_{d}", w * F.col(f"pctT_{d}") + (1 - w) * F.col(f"pctG_{d}"))
    units = percentile_within_type(units, "quality_shrunk", "q_norm")
    units = percentile_within_type(units, "value_raw", "v_norm")
    units = percentile_within_type(units, "n", "pop_norm")

    units = units.withColumn(
        "joya",
        ((F.col("v_norm") >= config.GEM_MIN_VALUE)
         & (F.col("q_norm") >= config.GEM_MIN_QUALITY)
         & (F.col("pop_norm") <= config.GEM_MAX_POPULARITY)).cast("int"),
    )
    return units, mu


def town_top_terms(df, towns):
    """TF-IDF por destino: cada Town es un 'documento'; idf penaliza palabras presentes en todos."""
    town_tokens = {tok for t in towns for tok in t.lower().replace("_", " ").split()}
    tokens = (
        df.select("Town", F.explode(F.split("text", " ")).alias("tok"))
        .filter(F.length("tok") >= 4)
        .filter(~F.col("tok").isin(*town_tokens))
    )
    tf = tokens.groupBy("Town", "tok").agg(F.count("*").alias("tf"))
    totals = tf.groupBy("Town").agg(F.sum("tf").alias("tot"))
    n_towns = totals.count()
    doc_freq = tf.groupBy("tok").agg(F.countDistinct("Town").alias("dfreq"))

    scored = (
        tf.join(totals, "Town").join(doc_freq, "tok")
        .filter(F.col("tf") >= config.MIN_TERM_COUNT)
        .withColumn(
            "tfidf",
            (F.col("tf") / F.col("tot")) * F.log((F.lit(n_towns) + 1) / (F.col("dfreq") + 1)),
        )
    )
    w = Window.partitionBy("Town").orderBy(F.col("tfidf").desc())
    top = scored.withColumn("rk", F.row_number().over(w)).filter(F.col("rk") <= config.TOP_TERMS)

    terms = {}
    for r in top.orderBy("Town", "rk").collect():
        terms.setdefault(r["Town"], []).append(r["tok"])
    return terms


def main():
    spark = (
        SparkSession.builder.master("local[*]").appName("FusionSensorial")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = load_reviews(spark).cache()
    print(f"Reseñas cargadas: {df.count():,}")

    units, mu = build_units(df)
    towns = [r["Town"] for r in df.select("Town").distinct().collect()]
    terms = town_top_terms(df, towns)

    profiles = []
    for r in units.orderBy("Town", "Type").collect():
        profiles.append({
            "town": r["Town"], "region": r["Region"], "type": r["Type"],
            "n": int(r["n"]),
            "pol_mean": round(r["pol_mean"], 3),
            "quality_shrunk": round(r["quality_shrunk"], 3),
            "q_norm": round(r["q_norm"], 4), "v_norm": round(r["v_norm"], 4),
            "pop_norm": round(r["pop_norm"], 4), "joya": bool(r["joya"]),
            "rate_barato": round(r["rate_valor_barato"], 4),
            "rate_caro": round(r["rate_valor_caro"], 4),
            # perfil turístico centrado en [-1, 1] (0 = mediana; mezcla percentil del Type y global)
            "t": {d: round((r[f"pct_{d}"] - 0.5) * 2, 4) for d in TOURISM_DIMS},
            "top_terms": terms.get(r["Town"], []),
        })

    config.PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump({"global_mean_polarity": round(mu, 4), "units": profiles}, f,
                  ensure_ascii=False, indent=1)
    print(f"Perfiles escritos: {len(profiles)} unidades (Town, Type) en {config.PROFILES_PATH}")
    spark.stop()


if __name__ == "__main__":
    main()
