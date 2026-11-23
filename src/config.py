"""Configuración central. Cada valor puede justificarse en la presentación."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Ruta del dataset. Por defecto: data/raw/. Se puede sobrescribir sin tocar el código:
#   export RESTMEX_CSV=/ruta/dentro/de/WSL/Rest-Mex_2025_train_clean.csv
DATA_PATH = Path(os.getenv("RESTMEX_CSV", ROOT / "data" / "raw" / "Rest-Mex_2025_train_clean.csv"))
PROFILES_PATH = ROOT / "data" / "processed" / "profiles.json"
PRESETS_PATH = ROOT / "data" / "presets.json"
MUSIC_MODEL_PATH = ROOT / "models" / "gtzan_mlp.joblib"

# --- Procesamiento (Spark) ---
MIN_REVIEWS = 150     # mínimo de reseñas por (Town, Type); ~percentil 10 real de los datos
PRIOR_WEIGHT = 200    # m del suavizado bayesiano de la polaridad: q = (n*media + m*mu)/(n+m)
VALUE_ALPHA = 0.01    # suavizado de V = log((barato+a)/(caro+a)); evita log(0) y división por 0
TYPE_PCT_WEIGHT = 0.5 # t = w*percentil_dentro_del_Type + (1-w)*percentil_global
TOP_TERMS = 8         # términos TF-IDF por destino (para explicar)
MIN_TERM_COUNT = 20   # frecuencia mínima de un término en un destino para ser candidato

# "Joya oculta": buen valor, buena calidad y poca popularidad (dentro de su Type)
GEM_MIN_VALUE = 0.60
GEM_MIN_QUALITY = 0.50
GEM_MAX_POPULARITY = 0.50

# --- Recomendación ---
BETA = 0.7            # peso de la afinidad sensorial frente a calidad/valor en el score final
FUSION_ALPHA = 0.5    # peso de la música frente al color al fusionar modalidades
